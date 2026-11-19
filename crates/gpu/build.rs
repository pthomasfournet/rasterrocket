//! Build script.
//!
//! Compiles compute kernels and emits link directives:
//! - `.cu` → PTX via `nvcc` for the CUDA backend (when any CUDA feature is on).
//! - `.slang` → SPIR-V via `slangc` for the Vulkan backend (when `vulkan` is on).
//! - `.slang` → HLSL via `slangc`, then DXIL via `dxc`, for the D3D12 backend
//!   (when `d3d12` is on; `build_support/dxil_build.rs`).
//! - Emits `cargo:rustc-link-{search,lib}` directives per active GPU feature.
//!
//! All artifacts land in `OUT_DIR` and are consumed by `include_str!` /
//! `include_bytes!` in `src/lib.rs` and the backend modules.
//!
//! Linux takes the CUDA toolkit from `/usr/local/cuda`; Windows from
//! `%CUDA_PATH%` (`bin\nvcc.exe`, `lib\x64\*.lib`), with nvcc's host compiler
//! `cl.exe` located from the Visual Studio installation.

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

include!("kernels/floor_div_params.rs");
include!("kernels/kernel_list.rs");

#[cfg(feature = "d3d12")]
#[path = "build_support/dxil_build.rs"]
mod dxil_build;

/// Emit a `cargo:rustc-link-search` directive for the first of `dirs` that
/// exists on the filesystem, then emit `cargo:rustc-link-lib=dylib={lib}`,
/// the library `feature` links.
///
/// If none of the directories exist, emit a build warning and let the linker
/// search its default paths (handles non-standard install locations).
fn link_lib_in_dir(dirs: &[PathBuf], lib: &str, feature: &str) {
    if let Some(dir) = dirs.iter().find(|d| d.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    } else {
        println!(
            "cargo:warning={feature}: none of the directories {dirs:?} that hold {lib} exists; \
             the linker searches its default paths"
        );
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

/// Slang profile for every kernel: baseline `spirv_1_5`, no subgroup
/// capabilities.
const SLANG_PROFILE: &str = "spirv_1_5";

/// Entry point slangc compiles for `kernel`, passed as `-entry name`; its
/// name need not match the file (e.g. `icc_clut.slang` → `icc_cmyk_clut`).
/// slangc renames the entry point to `main` in the SPIR-V.
fn slang_entry(kernel: &str) -> &'static str {
    match kernel {
        "icc_clut" => "icc_cmyk_clut",
        "blit_image" => "blit_image",
        "page_raster" | "page_raster_paths" | "page_raster_paths_poisoned" => "page_raster",
        "floor_div_probe" | "floor_div_probe_poisoned" => "floor_div_probe",
        _ => panic!("unknown kernel name: {kernel}"),
    }
}

/// The `.slang` file `kernel` is compiled from, and the preprocessor
/// macros it is compiled with besides [`slang_common_defines`]: a kernel
/// may be a variant of another's source.  `FLOOR_DIV_POISON` offsets
/// every floor division's first estimate so that its product wraps, for
/// the tests of the division's fault check.
fn slang_source(kernel: &str) -> (&str, &'static [&'static str]) {
    match kernel {
        "page_raster_paths" => ("page_raster", &["PAGE_PATHS"]),
        "page_raster_paths_poisoned" => ("page_raster", &["PAGE_PATHS", "FLOOR_DIV_POISON"]),
        "floor_div_probe_poisoned" => ("floor_div_probe", &["FLOOR_DIV_POISON"]),
        _ => (kernel, &[]),
    }
}

/// The preprocessor macros every Slang kernel is compiled with, as
/// `NAME=value`: the constants of `kernels/floor_div_params.rs`.
fn slang_common_defines() -> [String; 4] {
    [
        format!("STATUS_WORDS={STATUS_WORDS}"),
        format!("FLOOR_DIV_STEPS={FLOOR_DIV_STEPS}"),
        format!("FAULT_FLOOR_DIV={FAULT_FLOOR_DIV}"),
        format!("FAULT_FLOOR_DIV_WRAP={FAULT_FLOOR_DIV_WRAP}"),
    ]
}

/// Candidate directories for CUDA toolkit libraries on Linux, in preference
/// order.
///
/// First the generic symlink (`/usr/local/cuda`, which on Ubuntu is managed
/// by `update-alternatives` and points at whichever toolkit is selected),
/// then the legacy flat layout. A versioned tree such as `/usr/local/cuda-13.3`
/// is reached through the same symlink and is not enumerated.
/// Note: paths are x86-64-specific; GPU features are only supported on x86-64.
const CUDA_LIB_DIRS: &[&str] = &[
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/lib64",
];

/// Whether the crate is built for Windows.  The CUDA toolkit layout, nvcc's
/// host compiler and the static-library format follow the target; building
/// the CUDA features for another operating system than the host's is not
/// supported.
fn target_is_windows() -> bool {
    env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows")
}

/// Root of the CUDA toolkit on Windows: `CUDA_PATH`, which the toolkit
/// installer sets machine-wide to the most recently installed version.
fn windows_cuda_root() -> Option<PathBuf> {
    env::var_os("CUDA_PATH").map(PathBuf::from)
}

/// Candidate directories for CUDA toolkit libraries: [`CUDA_LIB_DIRS`] on
/// Linux, the import libraries in `%CUDA_PATH%\lib\x64` on Windows.
fn cuda_lib_dirs() -> Vec<PathBuf> {
    if target_is_windows() {
        windows_cuda_root()
            .map(|root| root.join("lib").join("x64"))
            .into_iter()
            .collect()
    } else {
        CUDA_LIB_DIRS.iter().map(PathBuf::from).collect()
    }
}

/// The nvcc to run: `NVCC` if set, else the toolkit's own
/// (`/usr/local/cuda/bin/nvcc` on Linux, `%CUDA_PATH%\bin\nvcc.exe` on
/// Windows) if it exists, else `nvcc` from `PATH`.
fn find_nvcc() -> PathBuf {
    if let Some(nvcc) = env::var_os("NVCC") {
        return nvcc.into();
    }
    let candidate = if target_is_windows() {
        windows_cuda_root().map(|root| root.join("bin").join("nvcc.exe"))
    } else {
        Some(PathBuf::from("/usr/local/cuda/bin/nvcc"))
    };
    candidate
        .filter(|p| p.exists())
        .unwrap_or_else(|| "nvcc".into())
}

/// The slangc to run: `SLANGC` if set, else the `LunarG` Vulkan SDK's
/// (`$VULKAN_SDK/bin/slangc`; `%VULKAN_SDK%\Bin\slangc.exe` on Windows) if
/// it exists, else `slangc` from `PATH`.  slangc runs on the build host, so
/// the executable suffix is the host's.
fn find_slangc() -> PathBuf {
    if let Some(slangc) = env::var_os("SLANGC") {
        return slangc.into();
    }
    env::var_os("VULKAN_SDK")
        .map(|sdk| {
            PathBuf::from(sdk)
                .join("bin")
                .join(format!("slangc{}", env::consts::EXE_SUFFIX))
        })
        .filter(|p| p.is_file())
        .unwrap_or_else(|| "slangc".into())
}

/// The host C++ compiler nvcc runs: extra nvcc arguments, and the
/// environment the compiler needs.  Empty where nvcc finds its host compiler
/// itself.
#[derive(Default)]
struct HostCompiler {
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
}

impl HostCompiler {
    /// On Windows nvcc needs `cl.exe` together with the MSVC `INCLUDE`,
    /// `LIB` and `PATH` environment.  `NVCC_CCBIN`, read by nvcc itself,
    /// names the compiler explicitly and disables the lookup; otherwise the
    /// MSVC installation is located as the `cc` crate locates it (a
    /// developer prompt's environment first, then the Visual Studio setup
    /// configuration) and passed to nvcc as `-ccbin` with its environment.
    /// On other targets nvcc finds its host compiler on `PATH`.
    fn detect() -> Self {
        if !target_is_windows() || env::var_os("NVCC_CCBIN").is_some() {
            return Self::default();
        }
        Self::detect_msvc()
    }

    #[cfg(windows)]
    fn detect_msvc() -> Self {
        let target = env::var("TARGET").expect("TARGET not set");
        let Some(cl) = cc::windows_registry::find_tool(&target, "cl.exe") else {
            println!(
                "cargo:warning=no MSVC cl.exe found for {target}; nvcc will search PATH. \
                 Install the Visual Studio Build Tools (C++ workload) or set NVCC_CCBIN."
            );
            return Self::default();
        };
        Self {
            args: vec!["-ccbin".into(), cl.path().into()],
            env: cl
                .get_envs()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
        }
    }

    #[cfg(not(windows))]
    fn detect_msvc() -> Self {
        println!("cargo:warning=a Windows target needs a Windows host to locate MSVC for nvcc");
        Self::default()
    }

    /// `nvcc` with this host compiler applied.
    fn apply<'a>(&self, nvcc: &'a mut Command) -> &'a mut Command {
        nvcc.args(&self.args)
            .envs(self.env.iter().map(|(k, v)| (k, v)))
    }
}

/// Directories searched for the nvJPEG2000 library.
///
/// On Linux, the CUDA 13 package installs under `libnvjpeg2k/13`; the
/// unversioned directory is the package's flat layout.  On Windows,
/// `NVJPEG2K_PATH` names the root of NVIDIA's redistributable archive
/// (`bin`, `include`, `lib`), then `cuda_lib_dirs`, the CUDA toolkit's
/// `lib\x64`, are searched.
fn nvjpeg2k_lib_dirs(cuda_lib_dirs: &[PathBuf]) -> Vec<PathBuf> {
    if target_is_windows() {
        env::var_os("NVJPEG2K_PATH")
            .map(|root| PathBuf::from(root).join("lib"))
            .into_iter()
            .chain(cuda_lib_dirs.iter().cloned())
            .collect()
    } else {
        vec![
            PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvjpeg2k/13"),
            PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvjpeg2k"),
        ]
    }
}

/// The directory holding `nvjpeg2k.h`: `NVJPEG2K_INCLUDE_DIR` if set, else
/// `/usr/include/libnvjpeg2k/13` on Linux (matching the link-time library)
/// and `%NVJPEG2K_PATH%\include` on Windows.  `None` leaves the header to
/// nvcc's own include path, the CUDA toolkit's.
fn nvjpeg2k_include_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("NVJPEG2K_INCLUDE_DIR") {
        return Some(dir.into());
    }
    if target_is_windows() {
        env::var_os("NVJPEG2K_PATH").map(|root| PathBuf::from(root).join("include"))
    } else {
        Some("/usr/include/libnvjpeg2k/13".into())
    }
}

/// Run `cmd`, the build step `what`, and panic with the tool's diagnostic
/// when it fails; `missing` is the hint added when the tool cannot be
/// started at all.
fn run_tool(cmd: &mut Command, what: &str, missing: &str) {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!(
            "failed to run {} for {what}: {e}{missing}",
            cmd.get_program().display()
        )
    });
    assert!(
        output.status.success(),
        "{} failed for {what} (status={}):\n{}",
        cmd.get_program().display(),
        output.status,
        tool_diagnostic(&output),
    );
}

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = PathBuf::from("kernels");

    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=shim/nvjpeg2k_shim.cpp");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVJPEG2K_PATH");
    println!("cargo:rerun-if-env-changed=NVJPEG2K_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let nvcc_path = find_nvcc();
    let nvcc = nvcc_path.as_os_str();
    let host = HostCompiler::detect();
    let cuda_lib_dirs = cuda_lib_dirs();
    let links_cuda = ["NVJPEG", "NVJPEG2K", "GPU_DESKEW"]
        .iter()
        .any(|feature| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some());
    if links_cuda && target_is_windows() && windows_cuda_root().is_none() {
        println!(
            "cargo:warning=CUDA_PATH is not set, so the CUDA libraries cannot be found; \
             install the CUDA toolkit (its installer sets CUDA_PATH) or set CUDA_PATH to its root"
        );
    }

    // nvJPEG: libnvjpeg.so ships with the CUDA toolkit.
    // On Linux this is provided by the NVIDIA driver, typically at
    // /usr/lib/x86_64-linux-gnu/libcuda.so.1 (driver-managed symlink).
    if env::var("CARGO_FEATURE_NVJPEG").is_ok() {
        link_lib_in_dir(&cuda_lib_dirs, "nvjpeg", "nvjpeg");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // nvJPEG2000: libnvjpeg2k.so lives in a non-standard path (not in the CUDA
    // toolkit tree) and requires cudart for cudaMalloc / cudaMemcpy2D.
    if env::var("CARGO_FEATURE_NVJPEG2K").is_ok() {
        link_lib_in_dir(&nvjpeg2k_lib_dirs(&cuda_lib_dirs), "nvjpeg2k", "nvjpeg2k");
        // cudart provides cudaMalloc / cudaFree / cudaMemcpy2D (runtime API).
        link_lib_in_dir(&cuda_lib_dirs, "cudart", "nvjpeg2k");
        // cuStreamSynchronize / cuCtxSetCurrent live in libcuda.so (driver).
        println!("cargo:rustc-link-lib=dylib=cuda");

        // Compile the C++ exception-boundary shim.  nvjpeg2k throws C++
        // exceptions on malformed codestreams; those cannot propagate through
        // Rust extern "C" FFI (undefined behaviour).  The shim wraps all
        // nvjpeg2k entry points that may throw in try/catch and maps any
        // exception to NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED (9).
        compile_nvjpeg2k_shim(&out_dir, nvcc, &host);
        println!("cargo:rustc-link-lib=static=nvjpeg2k_shim");
        // The shim uses C++ exception handling; link the C++ runtime.  MSVC
        // objects name their runtime libraries themselves (`/DEFAULTLIB`).
        if !target_is_windows() {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }

    // VA-API: libva.so.2 (core API) + libva-drm.so.2 (headless DRM connection).
    // Ships with Mesa (mesa-va-drivers) or intel-media-driver.
    //
    // The runtime libraries are in /usr/lib/x86_64-linux-gnu on Debian/Ubuntu but
    // the dev package (libva-dev) is not required — we link directly to the .so.2
    // versioned file using the `filename:` link syntax, which works without an
    // unversioned symlink.  This avoids a hard build-time dependency on libva-dev.
    if env::var("CARGO_FEATURE_VAAPI").is_ok() {
        const VA_DIRS: &[&str] = &["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib"];
        for dir in VA_DIRS {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
                break;
            }
        }
        // Link with explicit versioned filename so no unversioned symlink is required.
        println!("cargo:rustc-link-lib=dylib:+verbatim=libva.so.2");
        println!("cargo:rustc-link-lib=dylib:+verbatim=libva-drm.so.2");
    }

    // GPU deskew: CUDA NPP geometry library (nppiRotate_8u_C1R_Ctx lives in
    // libnppig; the stream context helpers nppSetStream/nppGetStreamContext live
    // in libnppc).  cudart and the driver are also required for cudaMalloc /
    // cudaMemcpy / cuStreamSynchronize.
    if env::var("CARGO_FEATURE_GPU_DESKEW").is_ok() {
        for lib in ["nppig", "nppc", "cudart"] {
            link_lib_in_dir(&cuda_lib_dirs, lib, "gpu-deskew");
        }
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // Allow `cfg(ptx_placeholder)` so `GpuCtx::init` can short-circuit
    // with a clear error when the build had no NVCC available and only
    // empty placeholders were written.
    println!("cargo:rustc-check-cfg=cfg(ptx_placeholder)");

    // PTX policy: compile real kernels whenever NVCC is available; write
    // empty placeholders only as a last-resort fallback for CPU-only CI
    // runners that don't have the CUDA toolkit.  Keying on a feature-flag
    // heuristic is wrong because rasterrocket-interp's `gpu-icc` feature
    // (which causes `GpuCtx::init` to be called at runtime) does not propagate
    // to the gpu crate's build environment, so a build with `gpu-icc` alone
    // would emit placeholders and produce a binary that crashes with
    // `CUDA_ERROR_INVALID_IMAGE` the first time `GpuCtx::init` runs.
    if nvcc_works(nvcc) {
        compile_cuda_kernels(&kernels_dir, &out_dir, nvcc, &host);
    } else {
        println!(
            "cargo:warning=NVCC unavailable ({}); writing placeholder PTX. \
             GpuCtx::init() will fail with a clear error if invoked at runtime.",
            nvcc_path.display()
        );
        println!("cargo:rustc-cfg=ptx_placeholder");
        write_placeholder_ptx(&out_dir);
    }

    // Slang→SPIR-V kernel compile, gated on the `vulkan` feature.  Independent
    // of the PTX path: a vulkan-only build (no CUDA features) still wants the
    // .spv artifacts.  Output is one .spv per kernel in OUT_DIR; the Vulkan
    // backend consumes them via include_bytes!.
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        compile_slang_kernels(&kernels_dir, &out_dir);
    }

    // Slang→HLSL→DXIL kernel compile for the D3D12 backend, on every host
    // so that the DXIL's tests run wherever the crate is tested.
    #[cfg(feature = "d3d12")]
    dxil_build::compile_dxil_kernels(&kernels_dir, &out_dir);
}

/// Probe whether `nvcc` is invokable.  Returns `false` when the binary
/// can't be executed (missing toolkit) so the build can fall back to
/// placeholder PTX rather than panic.
///
/// We intentionally invoke `--version` rather than just checking
/// existence: a stale symlink, a wrapper that requires unavailable
/// libraries, or a build-host without CUDA installed should all be
/// treated as "no NVCC".  On probe failure we emit a `cargo:warning`
/// with the diagnostic so the user can see *why* nvcc was rejected
/// (spawn error vs. non-zero exit) without `--verbose` builds.
fn nvcc_works(nvcc: &OsStr) -> bool {
    let shown = Path::new(nvcc).display();
    match Command::new(nvcc)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let stderr = stderr.trim();
            if stderr.is_empty() {
                println!(
                    "cargo:warning=nvcc probe `{shown} --version` exited with {} (no stderr output)",
                    o.status,
                );
            } else {
                println!(
                    "cargo:warning=nvcc probe `{shown} --version` exited with {}: {stderr}",
                    o.status,
                );
            }
            false
        }
        Err(e) => {
            println!("cargo:warning=nvcc probe `{shown} --version` could not be spawned: {e}");
            false
        }
    }
}

/// Compile every kernel's `.cu` source to PTX via `nvcc`.
///
/// `CUDA_ARCH` env var overrides the target architecture (e.g. `sm_120` for
/// Blackwell).  The default, `sm_75` (Turing), is the oldest architecture
/// CUDA 13 compiles for; the driver compiles its PTX for every newer device
/// when the module loads.
fn compile_cuda_kernels(kernels_dir: &Path, out_dir: &Path, nvcc: &OsStr, host: &HostCompiler) {
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_owned());

    for kernel in CUDA_KERNELS {
        let src = kernels_dir.join(format!("{kernel}.cu"));
        let ptx = out_dir.join(format!("{kernel}.ptx"));
        run_tool(
            host.apply(&mut Command::new(nvcc))
                .args(["--ptx", &format!("-arch={arch}"), "-O3", "--use_fast_math"])
                .arg("-o")
                .arg(&ptx)
                .arg(&src),
            &format!("{kernel}.cu (arch={arch})"),
            "",
        );
    }
}

/// A failed tool's diagnostic: its stderr and its stdout, each trimmed, the
/// empty ones left out.  nvcc writes some fatal errors (a host compiler
/// missing from `PATH`) to stdout alone.
fn tool_diagnostic(output: &std::process::Output) -> String {
    [&output.stderr, &output.stdout]
        .iter()
        .map(|bytes| String::from_utf8_lossy(bytes).trim().to_owned())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Write empty placeholder PTX files so the unconditional `include_str!` macros
/// in `src/lib.rs` compile successfully on CPU-only builds (Intel without CUDA,
/// ARM, CI runners without a GPU).  The placeholders are never loaded;
/// `GpuCtx::init()` fails with a CUDA error before any kernel is invoked.
///
/// Always overwritten (no existence check) because `OUT_DIR` is fresh on each
/// clean build and cargo guarantees this script only runs when inputs change.
fn write_placeholder_ptx(out_dir: &Path) {
    for kernel in CUDA_KERNELS {
        let ptx = out_dir.join(format!("{kernel}.ptx"));
        std::fs::write(&ptx, "")
            .unwrap_or_else(|e| panic!("failed to write placeholder {kernel}.ptx: {e}"));
    }
}

/// `slangc`, a slangc command, with the arguments that compile `kernel`
/// from `(source, defines)`, its [`slang_source`], added: the source file,
/// the entry point and the preprocessor macros.  The caller adds the
/// target and the output.
fn slang_args<'a>(
    slangc: &'a mut Command,
    kernels_dir: &Path,
    kernel: &str,
    (source, defines): (&str, &[&str]),
) -> &'a mut Command {
    let src = kernels_dir.join(format!("{source}.slang"));
    assert!(
        src.exists(),
        "slang source missing: {} (did you forget to add it alongside the .cu kernel?)",
        src.display()
    );
    slangc
        .args(["-entry", slang_entry(kernel)])
        .args(
            defines
                .iter()
                .map(|d| format!("-D{d}"))
                .chain(slang_common_defines().iter().map(|d| format!("-D{d}"))),
        )
        .arg(&src)
}

/// Compile every kernel's `.slang` source to SPIR-V via `slangc`.
///
/// Every kernel compiles against [`SLANG_PROFILE`].
///
/// The slangc run is [`find_slangc`]'s.
fn compile_slang_kernels(kernels_dir: &Path, out_dir: &Path) {
    let slangc = find_slangc();

    for kernel in CUDA_KERNELS.iter().chain(SLANG_ONLY_KERNELS) {
        let spv = out_dir.join(format!("{kernel}.spv"));
        let profile = SLANG_PROFILE;
        let source = slang_source(kernel);
        run_tool(
            slang_args(&mut Command::new(&slangc), kernels_dir, kernel, source)
                .args(["-target", "spirv", "-profile", profile])
                .arg("-o")
                .arg(&spv),
            &format!("{kernel} ({}.slang, profile={profile})", source.0),
            "; install the LunarG Vulkan SDK (it sets VULKAN_SDK), `apt install slang`, or \
             name slangc in SLANGC",
        );
    }
}

/// Compile `shim/nvjpeg2k_shim.cpp` into a static library in `OUT_DIR`:
/// `libnvjpeg2k_shim.a` (nvcc compiles, `ar` archives) or, for MSVC,
/// `nvjpeg2k_shim.lib` (`nvcc --lib`).  nvcc is the C++ compiler so the CUDA
/// headers `nvjpeg2k.h` includes are on the include path.  The MSVC build
/// enables C++ exception handling (`/EHsc`) and links the shared C runtime
/// (`/MD`), as Rust's MSVC targets do.
fn compile_nvjpeg2k_shim(out_dir: &Path, nvcc: &OsStr, host: &HostCompiler) {
    let shim_src = Path::new("shim/nvjpeg2k_shim.cpp");
    let include_arg = nvjpeg2k_include_dir().map(|dir| {
        let mut arg = OsString::from("-I");
        arg.push(dir);
        arg
    });
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    if target_is_windows() {
        let shim_lib = out_dir.join("nvjpeg2k_shim.lib");
        run_tool(
            host.apply(&mut Command::new(nvcc))
                .args(["-x", "c++", "-O2", "-Xcompiler", "/EHsc,/MD"])
                .args(&include_arg)
                .args(["--lib", "-o"])
                .arg(&shim_lib)
                .arg(shim_src),
            "nvjpeg2k_shim.lib",
            "",
        );
        return;
    }

    let shim_obj = out_dir.join("nvjpeg2k_shim.o");
    let shim_lib = out_dir.join("libnvjpeg2k_shim.a");
    run_tool(
        host.apply(&mut Command::new(nvcc))
            .args(["-x", "c++", "-O2"])
            .args(&include_arg)
            .args(["-c", "-o"])
            .arg(&shim_obj)
            .arg(shim_src),
        "nvjpeg2k_shim.o",
        "",
    );
    run_tool(
        Command::new("ar").arg("rcs").arg(&shim_lib).arg(&shim_obj),
        "libnvjpeg2k_shim.a",
        "; the archiver `ar` comes with binutils",
    );
}
