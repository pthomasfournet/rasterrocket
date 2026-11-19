// The shader compilers the `d3d12` build runs, their pinned versions and
// the commands that install them.  Pure functions, shared by `build.rs`
// (through `include!`) and the library's tests.

/// The slangc release every kernel is compiled with under `d3d12`:
/// `slangc -v` prints exactly this.
pub(crate) const SLANGC_VERSION: &str = "2026.20";

/// The DXC release the DXIL is compiled with: `dxc --version` prints it,
/// with a fourth component, inside parentheses.
pub(crate) const DXC_VERSION: &str = "1.9.2609";

/// Whether `output`, the text `slangc -v` printed, is [`SLANGC_VERSION`]
/// exactly; a build of a later commit (`2026.20-3-g…`) is another compiler.
pub(crate) fn slangc_version_ok(output: &str) -> bool {
    output.trim() == SLANGC_VERSION
}

/// Whether `output`, the text `dxc --version` printed, names
/// [`DXC_VERSION`]: the parenthesised full version (`(1.9.2609.5)`) begins
/// with it and a dot.
pub(crate) fn dxc_version_ok(output: &str) -> bool {
    output.contains(&format!("({DXC_VERSION}."))
}

/// The commands that install `tool` (`"slangc"` or `"dxc"`) at its pinned
/// version on a host of operating system `os` and architecture `arch`
/// (the values of `std::env::consts`), and the environment variable that
/// names the executable to the build.
///
/// A Windows host gets PowerShell commands.  slangc's release carries
/// builds for Linux and macOS on x86-64 and aarch64; DXC's carries a
/// Linux x86-64 build and a Windows one only, so on any other host the
/// hint states that DXC is built from its source at that release and
/// where the releases are.
///
/// # Panics
///
/// If `tool` is neither `"slangc"` nor `"dxc"`.
pub(crate) fn install_hint(tool: &str, os: &str, arch: &str) -> String {
    let dxc_releases = "https://github.com/microsoft/DirectXShaderCompiler/releases";
    let dxc_url = format!("{dxc_releases}/download/v{DXC_VERSION}");
    let slang_releases = "https://github.com/shader-slang/slang/releases";
    let slang_url = format!("{slang_releases}/download/v{SLANGC_VERSION}");
    let slang = format!("slang-{SLANGC_VERSION}");
    let commands = match (tool, os) {
        ("dxc", "windows") => format!(
            "$ProgressPreference = 'SilentlyContinue'\n\
             Invoke-WebRequest {dxc_url}/dxc_2026_09_29.zip -OutFile $env:TEMP\\dxc.zip\n\
             Expand-Archive $env:TEMP\\dxc.zip -DestinationPath C:\\tools\\dxc -Force\n\
             [Environment]::SetEnvironmentVariable('DXC', 'C:\\tools\\dxc\\bin\\x64\\dxc.exe', 'User')"
        ),
        ("dxc", "linux") if arch == "x86_64" => format!(
            "mkdir -p ~/.local/dxc ~/.local/bin && cd ~/.local/dxc\n\
             curl -LO {dxc_url}/linux_dxc_2026_09_28.x86_x64.tar.gz\n\
             tar xzf linux_dxc_2026_09_28.x86_x64.tar.gz && rm linux_dxc_2026_09_28.x86_x64.tar.gz\n\
             chmod +x bin/dxc bin/dxv\n\
             ln -sf ~/.local/dxc/bin/dxc ~/.local/bin/dxc\n\
             (or name dxc in DXC)"
        ),
        ("dxc", _) => format!(
            "DXC {DXC_VERSION} publishes builds for Linux x86-64 and Windows only \
             ({dxc_releases}/tag/v{DXC_VERSION}); on {os} {arch}, build DXC from the \
             source of that release, with its validator beside it, and name the dxc \
             executable in DXC"
        ),
        ("slangc", "windows") => format!(
            "$ProgressPreference = 'SilentlyContinue'\n\
             Invoke-WebRequest {slang_url}/{slang}-windows-x86_64.zip -OutFile $env:TEMP\\slang.zip\n\
             Expand-Archive $env:TEMP\\slang.zip -DestinationPath C:\\tools\\slang -Force\n\
             [Environment]::SetEnvironmentVariable('SLANGC', 'C:\\tools\\slang\\bin\\slangc.exe', 'User')"
        ),
        ("slangc", "linux" | "macos") if matches!(arch, "x86_64" | "aarch64") => format!(
            "mkdir -p ~/.local/slang ~/.local/bin && cd ~/.local/slang\n\
             curl -LO {slang_url}/{slang}-{os}-{arch}.tar.gz\n\
             tar xzf {slang}-{os}-{arch}.tar.gz && rm {slang}-{os}-{arch}.tar.gz\n\
             ln -sf ~/.local/slang/bin/slangc ~/.local/bin/slangc\n\
             (or name slangc in SLANGC)"
        ),
        ("slangc", _) => format!(
            "slangc {SLANGC_VERSION} publishes no build for {os} {arch} \
             ({slang_releases}/tag/v{SLANGC_VERSION}); build it from the source of \
             that release and name the slangc executable in SLANGC"
        ),
        _ => panic!("no install commands for tool {tool}"),
    };
    format!(
        "install {tool} {} ({os} {arch}):\n{commands}",
        tool_version(tool)
    )
}

/// The pinned version of `tool`.
fn tool_version(tool: &str) -> &'static str {
    if tool == "dxc" {
        DXC_VERSION
    } else {
        SLANGC_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_version_checks() {
        assert!(slangc_version_ok("2026.20\n"));
        assert!(!slangc_version_ok("2026.19\n"));
        assert!(!slangc_version_ok("2026.17-25-gc7954ebd1\n"));
        assert!(!slangc_version_ok("2026.20-3-gabcdef\n"));
        assert!(!slangc_version_ok(""));

        let linux = "libdxcompiler.so: 1.9(5512-01b62ad4)(1.9.2609.5)\n";
        let windows = "dxcompiler.dll: 1.9(5512-01b62ad4)(1.9.2609.5) - 1.9.2609.5 (01b62ad47)\n";
        assert!(dxc_version_ok(linux));
        assert!(dxc_version_ok(windows));
        assert!(!dxc_version_ok(&linux.replace("1.9.2609", "1.8.2505")));
        assert!(!dxc_version_ok(&windows.replace("1.9.2609", "1.8.2505")));
        assert!(!dxc_version_ok(""));

        let hint = install_hint("dxc", "windows", "x86_64");
        assert!(hint.contains("dxc"), "{hint}");
        assert!(
            hint.contains(
                "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/"
            ),
            "{hint}"
        );
        assert!(install_hint("dxc", "linux", "x86_64").contains("linux_dxc_"));
        for (os, arch) in [
            ("linux", "aarch64"),
            ("macos", "aarch64"),
            ("macos", "x86_64"),
        ] {
            let hint = install_hint("dxc", os, arch);
            assert!(!hint.contains("curl"), "{hint}");
            assert!(
                hint.contains(&format!("on {os} {arch}, build DXC")),
                "{hint}"
            );
            assert!(
                hint.contains("DirectXShaderCompiler/releases/tag/v1.9.2609"),
                "{hint}"
            );
        }
        let slangc = install_hint("slangc", "linux", "x86_64");
        assert!(
            slangc.contains(
                "https://github.com/shader-slang/slang/releases/download/v2026.20/\
                 slang-2026.20-linux-x86_64.tar.gz"
            ),
            "{slangc}"
        );
        for (os, arch) in [
            ("linux", "aarch64"),
            ("macos", "aarch64"),
            ("macos", "x86_64"),
        ] {
            let hint = install_hint("slangc", os, arch);
            assert!(
                hint.contains(&format!("slang-2026.20-{os}-{arch}.tar.gz")),
                "{hint}"
            );
        }
        assert!(install_hint("slangc", "freebsd", "x86_64").contains("publishes no build"));
        assert!(
            install_hint("slangc", "windows", "x86_64")
                .contains("slang-2026.20-windows-x86_64.zip")
        );
    }
}
