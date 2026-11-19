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
/// version, for a Windows host (PowerShell) when `windows` is true and a
/// Linux x86-64 host otherwise, and the environment variable that names
/// the executable to the build.
///
/// # Panics
///
/// If `tool` is neither `"slangc"` nor `"dxc"`.
pub(crate) fn install_hint(tool: &str, windows: bool) -> String {
    let dxc_url = format!(
        "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v{DXC_VERSION}"
    );
    let slang_url =
        format!("https://github.com/shader-slang/slang/releases/download/v{SLANGC_VERSION}");
    let slang = format!("slang-{SLANGC_VERSION}");
    let commands = match (tool, windows) {
        ("dxc", false) => format!(
            "mkdir -p ~/.local/dxc ~/.local/bin && cd ~/.local/dxc\n\
             curl -LO {dxc_url}/linux_dxc_2026_09_28.x86_x64.tar.gz\n\
             tar xzf linux_dxc_2026_09_28.x86_x64.tar.gz && rm linux_dxc_2026_09_28.x86_x64.tar.gz\n\
             chmod +x bin/dxc bin/dxv\n\
             ln -sf ~/.local/dxc/bin/dxc ~/.local/bin/dxc\n\
             (or name dxc in DXC)"
        ),
        ("dxc", true) => format!(
            "$ProgressPreference = 'SilentlyContinue'\n\
             Invoke-WebRequest {dxc_url}/dxc_2026_09_29.zip -OutFile $env:TEMP\\dxc.zip\n\
             Expand-Archive $env:TEMP\\dxc.zip -DestinationPath C:\\tools\\dxc -Force\n\
             [Environment]::SetEnvironmentVariable('DXC', 'C:\\tools\\dxc\\bin\\x64\\dxc.exe', 'User')"
        ),
        ("slangc", false) => format!(
            "mkdir -p ~/.local/slang ~/.local/bin && cd ~/.local/slang\n\
             curl -LO {slang_url}/{slang}-linux-x86_64.tar.gz\n\
             tar xzf {slang}-linux-x86_64.tar.gz && rm {slang}-linux-x86_64.tar.gz\n\
             ln -sf ~/.local/slang/bin/slangc ~/.local/bin/slangc\n\
             (or name slangc in SLANGC)"
        ),
        ("slangc", true) => format!(
            "$ProgressPreference = 'SilentlyContinue'\n\
             Invoke-WebRequest {slang_url}/{slang}-windows-x86_64.zip -OutFile $env:TEMP\\slang.zip\n\
             Expand-Archive $env:TEMP\\slang.zip -DestinationPath C:\\tools\\slang -Force\n\
             [Environment]::SetEnvironmentVariable('SLANGC', 'C:\\tools\\slang\\bin\\slangc.exe', 'User')"
        ),
        _ => panic!("no install commands for tool {tool}"),
    };
    format!("install {tool} {}:\n{commands}", tool_version(tool))
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

        let hint = install_hint("dxc", true);
        assert!(hint.contains("dxc"), "{hint}");
        assert!(
            hint.contains(
                "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/"
            ),
            "{hint}"
        );
        assert!(install_hint("dxc", false).contains("linux_dxc_"));
        let slangc = install_hint("slangc", false);
        assert!(
            slangc.contains(
                "https://github.com/shader-slang/slang/releases/download/v2026.20/\
                 slang-2026.20-linux-x86_64.tar.gz"
            ),
            "{slangc}"
        );
        assert!(install_hint("slangc", true).contains("slang-2026.20-windows-x86_64.zip"));
    }
}
