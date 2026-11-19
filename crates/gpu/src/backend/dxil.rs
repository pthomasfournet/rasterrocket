//! The DXIL of every kernel, compiled by `build.rs` (slangc to HLSL, DXC
//! to DXIL) and embedded in the binary, for the D3D12 backend.

include!("../../build_support/dxil_container.rs");

/// The kernels of the D3D12 backend, by the names `build.rs` compiles
/// them under.
pub(crate) const KERNELS: &[&str] = &["icc_clut", "blit_image", "page_raster", "page_raster_paths"];

/// Every kernel `build.rs` compiles to DXIL: [`KERNELS`], then the
/// test-only kernels (the floor division probe, and the variants whose
/// floor division is poisoned to fault).
#[cfg(test)]
const TEST_KERNELS: &[&str] = &[
    "icc_clut",
    "blit_image",
    "page_raster",
    "page_raster_paths",
    "floor_div_probe",
    "floor_div_probe_poisoned",
    "page_raster_paths_poisoned",
];

/// The DXIL container of `kernel`.
///
/// # Panics
///
/// If `kernel` is not one of [`KERNELS`] (or, in tests, of the test-only
/// kernels).
pub(crate) fn dxil(kernel: &str) -> &'static [u8] {
    match kernel {
        "icc_clut" => include_bytes!(concat!(env!("OUT_DIR"), "/icc_clut.dxil")),
        "blit_image" => include_bytes!(concat!(env!("OUT_DIR"), "/blit_image.dxil")),
        "page_raster" => include_bytes!(concat!(env!("OUT_DIR"), "/page_raster.dxil")),
        "page_raster_paths" => include_bytes!(concat!(env!("OUT_DIR"), "/page_raster_paths.dxil")),
        #[cfg(test)]
        "floor_div_probe" => include_bytes!(concat!(env!("OUT_DIR"), "/floor_div_probe.dxil")),
        #[cfg(test)]
        "floor_div_probe_poisoned" => {
            include_bytes!(concat!(env!("OUT_DIR"), "/floor_div_probe_poisoned.dxil"))
        }
        #[cfg(test)]
        "page_raster_paths_poisoned" => {
            include_bytes!(concat!(env!("OUT_DIR"), "/page_raster_paths_poisoned.dxil"))
        }
        _ => panic!("no DXIL is built for kernel {kernel}"),
    }
}

/// DXC's listing (`-Fc`) of the DXIL of `kernel`.
#[cfg(test)]
fn listing(kernel: &str) -> &'static str {
    match kernel {
        "icc_clut" => include_str!(concat!(env!("OUT_DIR"), "/icc_clut.ll")),
        "blit_image" => include_str!(concat!(env!("OUT_DIR"), "/blit_image.ll")),
        "page_raster" => include_str!(concat!(env!("OUT_DIR"), "/page_raster.ll")),
        "page_raster_paths" => include_str!(concat!(env!("OUT_DIR"), "/page_raster_paths.ll")),
        "floor_div_probe" => include_str!(concat!(env!("OUT_DIR"), "/floor_div_probe.ll")),
        "floor_div_probe_poisoned" => {
            include_str!(concat!(env!("OUT_DIR"), "/floor_div_probe_poisoned.ll"))
        }
        "page_raster_paths_poisoned" => {
            include_str!(concat!(env!("OUT_DIR"), "/page_raster_paths_poisoned.ll"))
        }
        _ => panic!("no DXIL listing is built for kernel {kernel}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The listing line that introduces the optional features a shader
    /// requires, and the one naming 64-bit integers.
    const REQUIRES: &str = "; Note: shader requires additional functionality:";
    const INT64: &str = ";       64-Bit integer";

    #[test]
    fn test_kernels_begin_with_the_production_kernels() {
        assert!(TEST_KERNELS.starts_with(KERNELS));
    }

    #[test]
    fn every_dxil_container_is_hashed() {
        for k in TEST_KERNELS {
            assert!(container_is_hashed(dxil(k)), "{k}");
        }
    }

    #[test]
    fn container_hash_check_reads_the_magic_and_the_digest() {
        let mut blob = [0_u8; 32];
        blob[..4].copy_from_slice(b"DXBC");
        assert!(!container_is_hashed(&blob), "zero digest");
        blob[19] = 1;
        assert!(container_is_hashed(&blob));
        assert!(!container_is_hashed(&blob[..19]), "truncated digest");
        blob[0] = b'X';
        assert!(!container_is_hashed(&blob), "wrong magic");
    }

    #[test]
    fn no_division_in_any_dxil_is_fast() {
        for k in TEST_KERNELS {
            for line in listing(k).lines().filter(|l| l.contains("fdiv")) {
                assert!(!line.contains("fast"), "{k}: {line}");
            }
        }
        assert!(listing("page_raster_paths").contains("fdiv"));
    }

    #[test]
    fn only_the_64_bit_kernels_declare_64_bit_integers() {
        let int64 = [
            "page_raster_paths",
            "page_raster_paths_poisoned",
            "floor_div_probe",
            "floor_div_probe_poisoned",
        ];
        for k in TEST_KERNELS {
            let lines: Vec<&str> = listing(k).lines().collect();
            let declared = lines.windows(2).any(|w| w == [REQUIRES, INT64]);
            assert_eq!(declared, int64.contains(k), "{k}");
            if !declared {
                assert!(
                    !lines.iter().any(|l| l.contains("64-Bit integer")),
                    "{k} names 64-bit integers outside the requirements block"
                );
            }
        }
    }
}
