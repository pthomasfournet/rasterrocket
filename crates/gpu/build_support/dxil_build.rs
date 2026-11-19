//! Compilation of every kernel to DXIL for the D3D12 backend (`d3d12`
//! feature), in two steps with pinned compilers: slangc writes HLSL and
//! its reflection, DXC compiles the HLSL to a signed DXIL container and
//! its listing.  The reflected registers and root-constant sizes are
//! checked against `kernels/d3d12_bindings.rs` before DXC runs.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use super::{
    CUDA_KERNELS, SLANG_ONLY_KERNELS, find_slangc, run_tool, slang_args, slang_entry, slang_source,
};

include!("tools.rs");
include!("dxil_container.rs");
include!("../kernels/d3d12_bindings.rs");

/// The DXIL shader model every kernel is compiled for: compute 6.0, the
/// lowest DXC emits.
const DXIL_PROFILE: &str = "cs_6_0";

/// The name of the constant buffer slangc builds from an entry point's
/// `uniform` scalars, which the reflection leaves unnamed; the HLSL calls
/// it `entryPointParams_0`.
const ENTRY_UNIFORMS: &str = "entryPointParams";

/// The DXC to run: `DXC` if set, else `dxc` from `PATH`.
fn find_dxc() -> PathBuf {
    env::var_os("DXC").map_or_else(|| "dxc".into(), PathBuf::from)
}

/// The version text `tool` (at `path`) prints for `flag`, from stdout and
/// stderr both.  Panics with the install commands when it cannot be run
/// or fails.
fn tool_version_text(path: &Path, flag: &str, tool: &str) -> String {
    let hint = install_hint(tool, cfg!(windows));
    let output = Command::new(path).arg(flag).output().unwrap_or_else(|e| {
        panic!(
            "failed to run {} {flag}: {e}; the d3d12 feature compiles the kernels with {tool} {}.\n{hint}",
            path.display(),
            tool_version(tool),
        )
    });
    let text = super::tool_diagnostic(&output);
    assert!(
        output.status.success(),
        "{} {flag} failed (status={}):\n{text}\nthe d3d12 feature compiles the kernels with {tool} {}.\n{hint}",
        path.display(),
        output.status,
        tool_version(tool),
    );
    text
}

/// Compile every kernel to `OUT_DIR/{kernel}.dxil`, with its DXC listing at
/// `{kernel}.ll`; the intermediate HLSL and reflection are
/// `{kernel}.hlsl` and `{kernel}.json`.
///
/// The slangc is [`find_slangc`]'s, the compiler of the SPIR-V, and must
/// print [`SLANGC_VERSION`]; the DXC is [`find_dxc`]'s and must print
/// [`DXC_VERSION`].
pub(super) fn compile_dxil_kernels(kernels_dir: &Path, out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=DXC");
    let slangc = find_slangc();
    let dxc = find_dxc();

    let slangc_version = tool_version_text(&slangc, "-v", "slangc");
    assert!(
        slangc_version_ok(&slangc_version),
        "slangc at {} printed version `{slangc_version}`; the d3d12 feature compiles the kernels \
         with slangc {SLANGC_VERSION} exactly.  The build runs SLANGC, else the Vulkan SDK's \
         slangc when VULKAN_SDK is set, else slangc from PATH: name a slangc \
         {SLANGC_VERSION} in SLANGC.\n{}",
        slangc.display(),
        install_hint("slangc", cfg!(windows)),
    );
    let dxc_version = tool_version_text(&dxc, "--version", "dxc");
    assert!(
        dxc_version_ok(&dxc_version),
        "dxc at {} printed version `{dxc_version}`; the d3d12 feature compiles the kernels with \
         DXC {DXC_VERSION}: name it in DXC.\n{}",
        dxc.display(),
        install_hint("dxc", cfg!(windows)),
    );
    println!(
        "cargo:warning=d3d12: slangc {} ({}), {} ({})",
        one_line(&slangc_version),
        slangc.display(),
        one_line(&dxc_version),
        dxc.display(),
    );

    let kernels: Vec<&str> = CUDA_KERNELS
        .iter()
        .chain(SLANG_ONLY_KERNELS)
        .copied()
        .collect();
    check_tables_cover(&kernels);
    for &kernel in &kernels {
        let out = |ext: &str| out_dir.join(format!("{kernel}.{ext}"));
        let (hlsl, json, dxil, listing) = (out("hlsl"), out("json"), out("dxil"), out("ll"));
        let source = slang_source(kernel);
        run_tool(
            slang_args(&mut Command::new(&slangc), kernels_dir, kernel, source)
                .args(["-target", "hlsl", "-o"])
                .arg(&hlsl)
                .arg("-reflection-json")
                .arg(&json),
            &format!("{kernel} ({}.slang, HLSL)", source.0),
            "",
        );
        let text = std::fs::read_to_string(&json)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", json.display()));
        let reflection: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: not JSON: {e}", json.display()));
        let (registers, constant_bytes) = reflect(kernel, &reflection);
        check_registers(kernel, &registers);
        check_constant_words(kernel, constant_bytes);

        run_tool(
            Command::new(&dxc)
                .args(["-T", DXIL_PROFILE, "-E", slang_entry(kernel)])
                .arg(&hlsl)
                .arg("-Fo")
                .arg(&dxil)
                .arg("-Fc")
                .arg(&listing),
            &format!("{kernel} (DXIL, {DXIL_PROFILE})"),
            "",
        );
        let blob =
            std::fs::read(&dxil).unwrap_or_else(|e| panic!("cannot read {}: {e}", dxil.display()));
        assert!(
            container_is_hashed(&blob),
            "{}: the DXIL container of {kernel} is not hashed; DXC signs it only when its \
             validator library (dxil.dll, libdxil.so) is installed beside it",
            dxil.display(),
        );
    }
}

/// `text` with its lines joined by `; `, for a `cargo:warning`, which
/// shows one line.
fn one_line(text: &str) -> String {
    text.lines().collect::<Vec<_>>().join("; ")
}

/// Panic unless [`KERNEL_REGISTERS`] and [`KERNEL_CONSTANT_WORDS`] each
/// hold one entry for every kernel of `kernels` and none for another.
fn check_tables_cover(kernels: &[&str]) {
    let mut want: Vec<&str> = kernels.to_vec();
    want.sort_unstable();
    for (table, mut names) in [
        (
            "KERNEL_REGISTERS",
            KERNEL_REGISTERS.iter().map(|e| e.0).collect::<Vec<_>>(),
        ),
        (
            "KERNEL_CONSTANT_WORDS",
            KERNEL_CONSTANT_WORDS.iter().map(|e| e.0).collect(),
        ),
    ] {
        names.sort_unstable();
        assert!(
            names == want,
            "{table} lists the kernels {names:?}; the build compiles {want:?} \
             (kernels/kernel_list.rs)"
        );
    }
}

/// A register found in the reflection.
#[derive(Debug, PartialEq, Eq)]
struct Reflected {
    name: String,
    class: RegisterClass,
    index: u32,
    space: u32,
}

/// The HLSL form of a register: `t1` in space 0 is `t1, space0`.
fn register_text(class: RegisterClass, index: u32, space: u32) -> String {
    let letter = match class {
        RegisterClass::Constants => 'b',
        RegisterClass::Srv => 't',
        RegisterClass::Uav => 'u',
    };
    format!("{letter}{index}, space{space}")
}

/// The elements of the JSON array `value`; none if it is not an array.
fn items(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

/// The `u32` at `key` of `object`, if present.
fn field_u32(object: &Value, key: &str) -> Option<u32> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| u32::try_from(v).expect("register numbers fit 32 bits"))
}

/// The registers slangc's reflection of `kernel` assigns, in its order,
/// and the size in bytes of its constant buffer (0 for none).
///
/// Global parameters are `parameters[i]`, each with its `binding`; a
/// global constant buffer's size is `type.elementVarLayout.binding.size`.
/// An entry point's parameters are `entryPoints[0].parameters[j]`: a
/// resource has a `binding`, a `uniform` scalar lies in the entry point's
/// constant buffer (`entryPoints[0].scope.binding`) at its `offset` and
/// `size`, and a system value has no binding.  An absent `space` is 0.
fn reflect(kernel: &str, reflection: &Value) -> (Vec<Reflected>, u32) {
    let mut found = Vec::new();
    let mut constant_bytes = 0;
    let mut add = |name: &str, binding: &Value| -> bool {
        let kind = binding["kind"].as_str().unwrap_or_default();
        let class = match kind {
            "constantBuffer" => RegisterClass::Constants,
            "shaderResource" => RegisterClass::Srv,
            "unorderedAccess" => RegisterClass::Uav,
            "uniform" => return false,
            _ => panic!(
                "{kernel}: parameter {name} has binding kind `{kind}`, which no root signature \
                 of this crate holds"
            ),
        };
        found.push(Reflected {
            name: name.to_owned(),
            class,
            index: field_u32(binding, "index").unwrap_or_else(|| {
                panic!("{kernel}: parameter {name} has no register index in the reflection")
            }),
            space: field_u32(binding, "space").unwrap_or(0),
        });
        true
    };

    for param in items(&reflection["parameters"]) {
        let name = param["name"].as_str().unwrap_or("?");
        let binding = &param["binding"];
        assert!(
            binding.is_object(),
            "{kernel}: global parameter {name} has no binding in the reflection"
        );
        if add(name, binding) && binding["kind"] == "constantBuffer" {
            constant_bytes = field_u32(&param["type"]["elementVarLayout"]["binding"], "size")
                .unwrap_or_else(|| {
                    panic!("{kernel}: the reflection gives no size for constant buffer {name}")
                });
        }
    }
    let entries = items(&reflection["entryPoints"]);
    assert!(
        entries.len() == 1,
        "{kernel}: the reflection has {} entry points, not one",
        entries.len()
    );
    let entry = &entries[0];
    let scope = &entry["scope"]["binding"];
    if scope.is_object() {
        let _ = add(ENTRY_UNIFORMS, scope);
    }
    let mut uniform_end = 0;
    for param in items(&entry["parameters"]) {
        let name = param["name"].as_str().unwrap_or("?");
        let binding = &param["binding"];
        if !binding.is_object() {
            continue;
        }
        if !add(name, binding) {
            let end = field_u32(binding, "offset").zip(field_u32(binding, "size"));
            let (offset, size) = end.unwrap_or_else(|| {
                panic!("{kernel}: uniform {name} has no offset and size in the reflection")
            });
            uniform_end = uniform_end.max(offset + size);
        }
    }
    if uniform_end > 0 {
        assert!(
            constant_bytes == 0,
            "{kernel}: both a global constant buffer and entry-point uniforms"
        );
        constant_bytes = uniform_end;
    }
    (found, constant_bytes)
}

/// Panic unless the [`KERNEL_REGISTERS`] entry of `kernel` is in its
/// documented order, each class's registers running from 0 in space 0
/// without a gap, and `found`, its reflected registers, are that entry,
/// parameter by parameter.
fn check_registers(kernel: &str, found: &[Reflected]) {
    let expected = KERNEL_REGISTERS
        .iter()
        .find(|(name, _)| *name == kernel)
        .unwrap_or_else(|| panic!("{kernel} has no entry in KERNEL_REGISTERS"))
        .1;
    assert!(
        expected
            .windows(2)
            .all(|w| (w[0].class, w[0].space, w[0].index) < (w[1].class, w[1].space, w[1].index)),
        "KERNEL_REGISTERS of {kernel} is not in order (constants, SRVs, UAVs, each by register)"
    );
    for class in [
        RegisterClass::Constants,
        RegisterClass::Srv,
        RegisterClass::Uav,
    ] {
        let of_class = expected.iter().filter(|r| r.class == class);
        for (n, r) in (0..).zip(of_class) {
            assert!(
                (r.index, r.space) == (n, 0),
                "{kernel}: parameter {} is {}; the registers of each class must run from 0 in \
                 space 0 without a gap, one descriptor range per class",
                r.name,
                register_text(r.class, r.index, r.space),
            );
        }
    }
    for want in expected {
        let shown = register_text(want.class, want.index, want.space);
        let got = found
            .iter()
            .find(|r| r.name == want.name)
            .unwrap_or_else(|| {
                panic!(
                    "{kernel}: parameter {}: KERNEL_REGISTERS has {shown}; the reflection has no \
                 such parameter",
                    want.name
                )
            });
        assert!(
            (got.class, got.index, got.space) == (want.class, want.index, want.space),
            "{kernel}: parameter {}: KERNEL_REGISTERS has {shown}; the reflection has {}",
            want.name,
            register_text(got.class, got.index, got.space),
        );
    }
    let extra: Vec<String> = found
        .iter()
        .filter(|r| !expected.iter().any(|w| w.name == r.name))
        .map(|r| format!("{} ({})", r.name, register_text(r.class, r.index, r.space)))
        .collect();
    assert!(
        extra.is_empty(),
        "{kernel}: parameters missing from KERNEL_REGISTERS: {}",
        extra.join(", ")
    );
}

/// Panic unless `bytes`, the reflected constant-buffer size of `kernel`,
/// is its [`KERNEL_CONSTANT_WORDS`] entry in 32-bit words.
fn check_constant_words(kernel: &str, bytes: u32) {
    let words = KERNEL_CONSTANT_WORDS
        .iter()
        .find(|(name, _)| *name == kernel)
        .unwrap_or_else(|| panic!("{kernel} has no entry in KERNEL_CONSTANT_WORDS"))
        .1;
    assert!(
        bytes == words * 4,
        "{kernel}: KERNEL_CONSTANT_WORDS has {words} words; the reflection's constant buffer is \
         {bytes} bytes"
    );
}
