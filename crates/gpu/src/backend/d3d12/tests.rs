//! The backend's adapter, buffer, recorder and kernel tests.

use windows::core::Interface as _;

use super::D3d12Backend;
use super::diag::{cpu_tier, warp_module};
use super::pipeline::KernelId;
use crate::backend::GpuBackend;
use crate::backend::page_device::PageDevice;
use crate::backend::select::DeviceSelector;

/// [`super::test_backend`] when its adapter has `Int64ShaderOps`; `None`
/// (the calling test is skipped) otherwise, which
/// `PDF_RASTER_REQUIRE_D3D12=1` turns into a failure.
fn test_backend_with_int64() -> Option<D3d12Backend> {
    let backend = super::test_backend()?;
    if backend.has_int64() {
        return Some(backend);
    }
    assert!(
        !std::env::var("PDF_RASTER_REQUIRE_D3D12").is_ok_and(|v| v == "1"),
        "PDF_RASTER_REQUIRE_D3D12=1, but the adapter lacks Int64ShaderOps"
    );
    eprintln!("skipping: the adapter lacks Int64ShaderOps");
    None
}

#[test]
fn d3d12_floor_div_matches_the_host_model() {
    use crate::backend::kernel_status::{FLOOR_DIV_STEPS, floor_div_cases};
    let Some(b) = test_backend_with_int64() else {
        return;
    };
    let cases = floor_div_cases(&mut crate::page::test_support::Rng(0x5eed));
    let (q, fault) = b.floor_div_probe(&cases, FLOOR_DIV_STEPS).unwrap();
    assert_eq!(fault, None);
    for (&(a, d), q) in cases.iter().zip(q) {
        assert_eq!(q, crate::page::path::floor_div(a, d), "{a} / {d}");
    }
}

#[test]
fn d3d12_floor_div_reports_an_exhausted_bound() {
    let Some(b) = test_backend_with_int64() else {
        return;
    };
    // Truncation toward zero leaves r = -1: one correction step is needed.
    let (_, fault) = b.floor_div_probe(&[(-1, 3)], 0).unwrap();
    assert_eq!(fault, Some((-1, 3)));
}

#[test]
fn d3d12_floor_div_reports_an_exhausted_upward_bound() {
    use crate::backend::kernel_status::{FAULT_FLOOR_DIV, FLOOR_DIV_STEPS, floor_div_fault};
    let Some(b) = test_backend_with_int64() else {
        return;
    };
    // Outside the domain: with |a / b| near 2^61 the second estimate
    // divides a remainder of about 2^37, whose float rounding leaves
    // r = 3 123 above the quotient after it, past 16 upward steps.
    let pair = (2_404_050_171_065_969_715, 1);
    let (_, status) = b
        .floor_div_probe_on(KernelId::FloorDivProbe, &[pair], FLOOR_DIV_STEPS)
        .unwrap();
    assert_eq!(status[0], FAULT_FLOOR_DIV);
    assert_eq!(floor_div_fault(&status), Some(pair));
}

/// The floor-division probe binds its status at `u0` and its
/// quotients at `u1`; a binding taken from the tables puts each
/// where the kernel reads it.
#[test]
fn floor_div_probe_binds_from_the_register_tables() {
    use crate::backend::kernel_status::{FLOOR_DIV_STEPS, STATUS_BYTES};

    let backend = D3d12Backend::warp().expect("the WARP adapter opens");
    if !backend.has_int64() {
        eprintln!("skipping: WARP lacks Int64ShaderOps");
        return;
    }
    // Operands within floor division's domain: `0 < b < 2^32`,
    // `|a / b| < 2^40`.
    let pairs: [(i64, i64); 5] = [
        (7, 2),
        (-7, 2),
        (0, 3),
        (-1, (1 << 32) - 1),
        (-(1 << 40) - 5, 256),
    ];
    let floor = |a: i64, b: i64| {
        let q = a / b;
        if a % b != 0 && (a < 0) != (b < 0) {
            q - 1
        } else {
            q
        }
    };
    let words: Vec<u8> = pairs
        .iter()
        .copied()
        .flat_map(<[i64; 2]>::from)
        .flat_map(i64::to_le_bytes)
        .collect();
    let n = u32::try_from(pairs.len()).expect("five pairs");
    let mut input = backend.alloc_device(words.len()).expect("input");
    backend.upload(&mut input, &words).expect("upload");
    let mut status = backend.alloc_device_zeroed(STATUS_BYTES).expect("status");
    let mut quotients = backend.alloc_device(pairs.len() * 8).expect("quotients");
    backend
        .run_page(|b| {
            b.recorder.dispatch(
                KernelId::FloorDivProbe,
                &[(&input, 4)],
                &[(&mut status, 4), (&mut quotients, 4)],
                &[n, FLOOR_DIV_STEPS],
                (n.div_ceil(64), 1, 1),
            )
        })
        .expect("probe page");
    let mut out = vec![0; pairs.len() * 8];
    backend.download(&quotients, &mut out).expect("quotients");
    let got: Vec<i64> = out
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect();
    let want: Vec<i64> = pairs.iter().map(|&(a, b)| floor(a, b)).collect();
    assert_eq!(got, want);
    let mut words = [0xFF; STATUS_BYTES];
    backend.download(&status, &mut words).expect("status");
    assert_eq!(words, [0; STATUS_BYTES], "the probe recorded a fault");
}

/// The reference count of `object`.
fn refs(object: &windows::core::IUnknown) -> u32 {
    let vtable = object.vtable();
    // SAFETY: `object` is live; the reference added is released at
    // once, and `Release` returns the count left.
    unsafe {
        let _added = (vtable.AddRef)(object.as_raw());
        (vtable.Release)(object.as_raw())
    }
}

/// Stranded commands refuse a new list and never release what the
/// list references, even when dropped; released commands do.
#[test]
fn stranded_commands_keep_their_references() {
    use super::commands::Commands;

    let backend = D3d12Backend::warp().expect("the WARP adapter opens");
    let buf = backend.alloc_device(64).expect("alloc_device");
    let unknown: windows::core::IUnknown = buf.resource().clone().into();
    let base = refs(&unknown);

    let mut released = Commands::new(&backend.device.device).expect("commands");
    released.begin().expect("begin");
    released.hold(buf.resource());
    assert_eq!(refs(&unknown), base + 1);
    released.discard();
    assert_eq!(refs(&unknown), base, "discard releases the hold");
    drop(released);

    let mut stranded = Commands::new(&backend.device.device).expect("commands");
    stranded.begin().expect("begin");
    stranded.hold(buf.resource());
    stranded.strand();
    stranded.release();
    stranded.discard();
    assert!(stranded.begin().is_err(), "stranded commands are not reset");
    drop(stranded);
    assert_eq!(refs(&unknown), base + 1, "the stranded hold is leaked");
}

/// A page whose own fence cannot be waited on keeps everything its
/// list references, through `wait_page` and the backend's drop, and
/// no further page begins; a page dropped unwaited does the same
/// when the queue cannot be drained.
#[test]
fn unconfirmed_pages_keep_their_references() {
    for waited in [true, false] {
        let backend = D3d12Backend::warp().expect("the WARP adapter opens");
        let mut buf = backend.alloc_device(64).expect("alloc_device");
        let unknown: windows::core::IUnknown = buf.resource().clone().into();
        let base = refs(&unknown);
        backend.begin_page().expect("begin_page");
        backend.record_zero_buffer(&mut buf, 64).expect("fill");
        let fence = backend.submit_page().expect("submit_page");
        backend.submitter.fail_waits();
        if waited {
            assert!(backend.wait_page(fence).is_err(), "the wait fails");
            let err = backend.begin_page().expect_err("the recorder is lost");
            assert!(err.to_string().contains("could not be confirmed"), "{err}");
            backend.abandon_page();
        }
        drop(backend);
        assert_eq!(
            refs(&unknown),
            base + 1,
            "waited {waited}: the list's hold on the buffer was released"
        );
    }
}

/// The padded upload writes zeros after the bytes up to a whole
/// word, and refuses a padded length past the buffer.
#[test]
fn d3d12_padded_upload_zeroes_the_tail() {
    let backend = D3d12Backend::warp().expect("the WARP adapter opens");
    let mut buf = backend.alloc_device(8).expect("alloc_device");
    backend.upload(&mut buf, &[0xFF; 8]).expect("fill");
    backend
        .upload_word_padded(&buf, &[1, 2, 3, 4, 5])
        .expect("upload_word_padded");
    let mut out = [0; 8];
    backend.download(&buf, &mut out).expect("download");
    assert_eq!(out, [1, 2, 3, 4, 5, 0, 0, 0]);
    let err = backend
        .upload_word_padded(&buf, &[0; 9])
        .expect_err("9 bytes pad to 12, past the 8-byte buffer");
    assert!(err.to_string().contains("copy of 12 bytes"), "{err}");
}

/// Every kernel's root signature and pipeline state build.
#[test]
fn every_kernel_pipeline_builds_on_warp() {
    let backend = D3d12Backend::warp().expect("the WARP adapter opens");
    for &k in KernelId::ALL {
        if matches!(
            k,
            KernelId::PageRasterPaths | KernelId::FloorDivProbe | KernelId::PageRasterPathsPoisoned
        ) && !backend.has_int64()
        {
            continue;
        }
        backend
            .recorder
            .pipeline(k)
            .unwrap_or_else(|e| panic!("{}: {e}", k.label()));
    }
}

#[test]
fn warp_opens_and_logs_its_module() {
    let backend = D3d12Backend::warp().expect("the WARP adapter opens");
    assert!(backend.is_warp(), "{}", backend.device_name());
    let (path, version) = warp_module().expect("d3d10warp.dll is loaded");
    let file = path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    assert!(
        file.eq_ignore_ascii_case("d3d10warp.dll"),
        "{}",
        path.display()
    );
    eprintln!(
        "WARP {version} from {} on a CPU with {}; adapter {} \"{}\", int64 {}",
        path.display(),
        cpu_tier(),
        backend.device_index(),
        backend.device_name(),
        backend.has_int64()
    );
}

#[test]
fn hardware_adapters_open_by_index_and_automatically() {
    let gpus = match D3d12Backend::usable_gpus() {
        Ok(gpus) => gpus,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let auto = D3d12Backend::open(None).expect("the automatic choice opens");
    assert_eq!(auto.device_index(), gpus[0]);
    assert!(!auto.is_warp());
    for &i in &gpus {
        let sel: DeviceSelector = i.to_string().parse().expect("selector");
        let backend = D3d12Backend::open(Some(&sel)).expect("a usable adapter opens");
        assert_eq!(backend.device_index(), i);
        assert!(!backend.is_warp());
        eprintln!(
            "adapter {i} \"{}\", int64 {}",
            backend.device_name(),
            backend.has_int64()
        );
    }
}
