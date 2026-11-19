# Windows driver of the cross-OS benchmark: builds the binaries, records the
# machine state, and runs the rows of tests/compare/os_bench_rows.tsv through
# the os_bench harness under fixed CPU affinity.  Windows PowerShell 5.1.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\compare\os_bench.ps1
#       -Out DIR [-Group G]... [-Only ROW [-Label NAME]] [-Reps K] [-Smoke]
#       [-RamRoot R:\] [-DefenderExcluded] [-GpuFeatures LIST] [-DryRun]
#       [-Launch]
#
#   -Out DIR           result directory, absent or empty.
#   -Group G           run the rows of G (library, startup, disk, ram, avx2,
#                      gpu); repeatable, or a comma-separated list; default
#                      every group.
#   -Only ROW          run ROW alone (overrides -Group).
#   -Label NAME        with -Only: the row's samples carry NAME.
#   -Reps K            repetitions per row (default 3).
#   -Smoke             one document per row, one repetition, and none of the
#                      machine-state checks (power scheme, clean tree).  The
#                      RUSTFLAGS check stays; a RAM root that fails its check
#                      skips the ram rows, with a note.
#   -RamRoot R:\       root drive of the ram rows' --ram-path directories.  It
#                      must be the root of a dedicated RAM disk of at most
#                      4 GiB: not the system drive nor the drive of DIR or of
#                      the repository, writable, and holding nothing but this
#                      driver's rr-os-bench directory (and the file system's
#                      own System Volume Information and $RECYCLE.BIN).
#                      Without it the ram rows are skipped, each with a line
#                      in skipped.txt.
#   -DefenderExcluded  with -Only ROW: run ROW-excl when the rows file has
#                      it, else ROW; add the Defender exclusions of DIR and
#                      rrocket.exe that are not present already, remove them
#                      when the run ends, and record defender_excl true.
#                      Needs an elevated session.
#   -GpuFeatures L     features of the gpu build (default
#                      nvjpeg,gpu-icc,gpu-deskew,cache,vulkan).  A list
#                      without a CUDA feature skips the --backend cuda row,
#                      one without vulkan the --backend vulkan rows.
#   -DryRun            print every command (builds, affinity prefixes,
#                      harness calls; with -Launch the task commands) and exit
#                      0; nothing is built, run or written.
#   -Launch            write the other arguments to DIR\launch-params.txt and
#                      start this script on them as the scheduled task
#                      rr-os-bench, in the session of the user logged in at
#                      the console, then return.  The task runs with highest
#                      privileges when -DefenderExcluded is given.  The user
#                      name is read at run time and written to no file; /f
#                      replaces a task of that name left by an earlier launch.
#
# Rows whose os column is linux never run here, nor do rows whose os column is
# windows without -DefenderExcluded.  The S-version row runs first in every
# run, whatever the selection, since the summary needs it.  Rows of the avx2
# group run twice: on HEAD, and as ROW-pre on a scratch worktree
# (target\os-bench\pre-avx2) with commits a5cfd530 and 205be502 reverted; a
# revert that does not apply cleanly stops the run.  A worktree left by an
# earlier run at the same HEAD, unmodified, is reused.  Rows of the gpu group
# run on a build with -GpuFeatures in target\os-bench\gpu (CUDA_ARCH=sm_75);
# the --backend cuda row also needs a GPU driver whose nvidia-smi version is
# 580 or later.  Each skipped row adds a `skipped ROW: reason` line to
# DIR\skipped.txt.
#
# Affinity: each harness call runs as `start /b /wait /affinity MASK` under
# cmd, with RAYON_NUM_THREADS=N: a 1-thread row has mask 4 (logical CPU 2),
# an N-thread row the mask of CPUs 0..N-1 (FFF for twelve).  The start-up
# rows (S-version, S-page, G-cuda-start, G-vulkan-start) are 1-thread rows:
# the process, with every runtime and driver thread it starts, shares CPU 2,
# so a G row less S-page is the backend's initialisation on one core.  Disk
# rows write under DIR\scratch (--no-ram and a path prefix, counted with
# --expect-pages); ram rows write to --ram-path under the RAM root and spill
# to an empty DIR\scratch directory.  After the rows, every row
# `os_bench noisy` lists is run again with 5 repetitions into rerun.tsv,
# which supersedes its samples.
#
# Prerequisites: cargo and rustc of the repository toolchain (MSVC target),
# git, nasm (on PATH or in %ProgramFiles%\NASM, added to PATH for the
# builds), and the fixtures named in the rows file under tests\fixtures\
# (gitignored).  RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS and CARGO_BUILD_RUSTFLAGS
# must be unset, so that the repository's -C target-cpu=native applies.  A
# full run also requires the High performance or Ultimate Performance power
# scheme and a working tree equal to HEAD; each failed check prints its
# remedy, and the system change needs the owner's go-ahead:
#   powercfg /setactive SCHEME_MIN
# Large pages need SeLockMemoryPrivilege, which an unprivileged process does
# not hold by default, so they are off without a check and no env key records
# them.  Windows Update, the Defender scan schedule and indexing of the bench
# volume are set by hand before a run; the driver records the power scheme,
# hardware-accelerated GPU scheduling and Defender real-time protection.
#
# DIR receives the files of the Linux driver:
#   env.tsv      key<TAB>value machine state, from a fixed key set
#   results.tsv  one harness sample per line
#   rerun.tsv    5-repetition re-runs of noisy rows (when any)
#   pages.tsv    per-page render times of the rows that keep them
#   build.log    output of every build step
#   skipped.txt  rows not run and why (when any)
#   done.flag    written last, on success
#   failed.flag  instead, on any error, interruption or failed cleanup step
#                (removing the RAM directory or a Defender exclusion) once DIR
#                is taken (a usage error or a non-empty DIR exits 2 without
#                it), holding the first failing step and command
# and also harness.log, the harness's standard error, whose last lines a
# failing harness call adds to failed.flag.  Every file is ASCII without a
# byte-order mark.  Ctrl-C stops the running child process tree and then the
# driver; a task ended by schtasks /end is killed outright and leaves neither
# flag.  Paths in build.log, harness.log and failed.flag may name the home
# directory: none of them is published.  env.tsv, results.tsv, rerun.tsv,
# pages.tsv and skipped.txt hold no host or user name.

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$Here = $PSScriptRoot
$Root = [System.IO.Path]::GetFullPath((Join-Path $Here '..\..'))
$RowsFile = Join-Path $Here 'os_bench_rows.tsv'
$Ascii = New-Object System.Text.ASCIIEncoding
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

$RamGibLimit = 4
$AvxCommits = @('a5cfd530', '205be502')
$CudaFeatures = @('nvjpeg', 'nvjpeg2k', 'gpu-icc', 'gpu-deskew', 'cache')
$CudaDriverFloor = 580
$TaskName = 'rr-os-bench'
$ParamsName = 'launch-params.txt'
$TaskCommandLimit = 261
$PowerSchemes = @{
    '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c' = 'High performance'
    'e9a42b02-d5df-448d-aa00-03f14749eb61' = 'Ultimate Performance'
}
$PowerRemedy = "with the owner's go-ahead: powercfg /setactive SCHEME_MIN"

# ---- output -------------------------------------------------------------------

function Write-Err([string]$Text) {
    [Console]::Error.WriteLine($Text)
}

function Write-Line([string]$Text) {
    [Console]::Out.WriteLine($Text)
}

function Show-Usage([int]$Code) {
    $print = $false
    foreach ($line in [System.IO.File]::ReadAllLines($PSCommandPath)) {
        if ($line -like '# Usage:*') { $print = $true }
        if ($line -like '# Rows whose*') { break }
        if ($print) { Write-Err ($line -replace '^# ?', '') }
    }
    exit $Code
}

function Exit-Usage([string]$Text) {
    Write-Err "os_bench.ps1: $Text"
    Show-Usage 2
}

# ---- arguments ----------------------------------------------------------------

# Read-Option ARGV: the options as a hashtable.  Named parameters are
# parsed here rather than by the binder, which refuses a repeated -Group.
function Read-Option([string[]]$Argv) {
    $o = @{
        Out = ''; Group = New-Object System.Collections.Generic.List[string]
        Only = ''; Label = ''; Reps = '3'; Smoke = $false; RamRoot = ''
        DefenderExcluded = $false; GpuFeatures = 'nvjpeg,gpu-icc,gpu-deskew,cache,vulkan'
        DryRun = $false; Launch = $false; ParamsFile = ''
    }
    $valued = @('Out', 'Group', 'Only', 'Label', 'Reps', 'RamRoot', 'GpuFeatures', 'ParamsFile')
    $switches = @('Smoke', 'DefenderExcluded', 'DryRun', 'Launch')
    $i = 0
    while ($i -lt $Argv.Length) {
        $a = $Argv[$i]
        if ($a -in @('-h', '-Help', '-?', '--help')) { Show-Usage 0 }
        if ($a -notmatch '^-([A-Za-z]+)$') { Exit-Usage "unexpected argument: $a" }
        $name = $Matches[1]
        $key = @($valued + $switches) | Where-Object { $_ -eq $name } | Select-Object -First 1
        if (-not $key) { Exit-Usage "unknown parameter: $a" }
        if ($key -in $switches) {
            $o[$key] = $true
            $i += 1
            continue
        }
        if ($i + 1 -ge $Argv.Length) { Exit-Usage "$a needs a value" }
        $v = $Argv[$i + 1]
        if ($key -eq 'Group') {
            foreach ($g in $v.Split(',')) {
                if ($g -ne '') { $o.Group.Add($g) }
            }
        } else {
            $o[$key] = $v
        }
        $i += 2
    }
    return $o
}

$Opt = Read-Option $args
if ($Opt.ParamsFile -ne '') {
    if ($args.Length -ne 2) { Exit-Usage '-ParamsFile takes no other argument' }
    $ParamsPath = [System.IO.Path]::GetFullPath($Opt.ParamsFile)
    if (-not (Test-Path -LiteralPath $ParamsPath -PathType Leaf)) { Exit-Usage "$ParamsPath does not exist" }
    $Opt = Read-Option ([System.IO.File]::ReadAllLines($ParamsPath))
    if ($Opt.Out -ne '' -or $Opt.ParamsFile -ne '' -or $Opt.Launch) {
        Exit-Usage "$ParamsPath may not hold -Out, -ParamsFile or -Launch"
    }
    $Opt.Out = Split-Path -Parent $ParamsPath
} else {
    $ParamsPath = ''
}

if ($Opt.Out -eq '') { Exit-Usage '-Out is required' }
if ($Opt.Reps -notmatch '^[1-9][0-9]*$') { Exit-Usage "-Reps must be a positive integer, got: $($Opt.Reps)" }
if ($Opt.Label -ne '' -and $Opt.Only -eq '') { Exit-Usage '-Label needs -Only' }
if ($Opt.DefenderExcluded -and $Opt.Only -eq '') { Exit-Usage '-DefenderExcluded needs -Only' }
if ($Opt.Label -match '[\s"%]') { Exit-Usage "-Label may not hold white space, quotes or %: $($Opt.Label)" }
$Reps = [int]$Opt.Reps
if ($Opt.Smoke) { $Reps = 1 }
$Dry = [bool]$Opt.DryRun
$Smoke = [bool]$Opt.Smoke
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Opt.Out).TrimEnd('\')

$HeadRr = Join-Path $Root 'target\release\rrocket.exe'
$HeadBench = Join-Path $Root 'target\release\os_bench.exe'
$PreDir = Join-Path $Root 'target\os-bench\pre-avx2'
$PreTarget = Join-Path $Root 'target\os-bench\pre-avx2-target'
$PreBench = Join-Path $PreTarget 'release\os_bench.exe'
$PreStamp = Join-Path $Root 'target\os-bench\pre-avx2.tree'
$GpuTarget = Join-Path $Root 'target\os-bench\gpu'
$GpuRr = Join-Path $GpuTarget 'release\rrocket.exe'
$Scratch = Join-Path $Out 'scratch'
$BuildLog = Join-Path $Out 'build.log'
$HarnessLog = Join-Path $Out 'harness.log'

# ---- failure reporting ----------------------------------------------------------

$Step = 'argument checks'
$FlagOk = $false    # failed.flag may be written once DIR is the run's own.
$Finished = $false
$Child = $null      # the running child process, stopped on interruption.
$RamMade = $false   # the RAM directory exists and is removed at the end.
$ExclPaths = @()    # Defender exclusions this run added, removed at the end.
$ExclProcs = @()
$CleanupFailed = $false

# Hide-Path TEXT: TEXT with the repository and home paths shortened.
function Hide-Path([string]$Text) {
    $t = [regex]::Replace($Text, [regex]::Escape("$Root\"), '', 'IgnoreCase')
    if ($HOME -and $HOME.Length -gt 3) {
        $t = [regex]::Replace($t, [regex]::Escape($HOME), '~', 'IgnoreCase')
    }
    return $t
}

# Write-Flag MESSAGE: failed.flag holding MESSAGE, unless one exists: the
# first report is kept.
function Write-Flag([string]$Message) {
    $flag = Join-Path $Out 'failed.flag'
    if ($FlagOk -and -not (Test-Path -LiteralPath $flag)) {
        [System.IO.File]::WriteAllText($flag, "$Message`n", $Ascii)
    }
}

# Exit-Failure MESSAGE [REMEDY]: a check failed; the remedy is reported with it.
function Exit-Failure([string]$Message, [string]$Remedy = '') {
    $text = "${Step}: $Message"
    if ($Remedy -ne '') { $text += " (remedy: $Remedy)" }
    $e = New-Object System.Exception $text
    $e.Data['os_bench'] = $true
    throw $e
}

# Invoke-Cleanup WHAT BLOCK: BLOCK, whose failure is reported, written to
# failed.flag and remembered, but stops nothing else.
function Invoke-Cleanup([string]$What, [scriptblock]$Block) {
    try {
        & $Block
    } catch {
        $script:CleanupFailed = $true
        $msg = Hide-Path "cleanup: ${What}: $($_.Exception.Message)"
        Write-Err "os_bench.ps1: $msg"
        Write-Flag $msg
    }
}

# ---- process execution --------------------------------------------------------------

# ConvertTo-Arg TEXT [-ForCmd]: TEXT quoted for a Windows command line.  Under
# cmd, the characters cmd would expand inside quotes are refused.
function ConvertTo-Arg([string]$Text, [switch]$ForCmd) {
    if ($ForCmd -and $Text -match '["%\r\n]') { Exit-Failure "argument holds a character cmd would interpret: $Text" }
    $special = '[\s"]'
    if ($ForCmd) { $special = '[\s"&|<>()^,;=]' }
    if ($Text -ne '' -and $Text -notmatch $special) { return $Text }
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    $slashes = 0
    foreach ($c in $Text.ToCharArray()) {
        if ($c -eq '\') { $slashes += 1; continue }
        if ($c -eq '"') {
            [void]$sb.Append('\', 2 * $slashes + 1).Append('"')
        } else {
            [void]$sb.Append('\', $slashes).Append($c)
        }
        $slashes = 0
    }
    [void]$sb.Append('\', 2 * $slashes).Append('"')
    return $sb.ToString()
}

function Join-Argument([string[]]$Argv, [switch]$ForCmd) {
    return (@($Argv | ForEach-Object { ConvertTo-Arg $_ -ForCmd:$ForCmd }) -join ' ')
}

# Format-Env ENV: the variables as NAME=VALUE prefixes of a shown command,
# with this process's own PATH shown as %PATH%.
function Format-Env([hashtable]$Env) {
    $s = ''
    foreach ($k in @($Env.Keys | Sort-Object)) {
        $v = [string]$Env[$k]
        if ($k -eq 'PATH' -and $v.EndsWith(";$env:PATH")) { $v = $v.Substring(0, $v.Length - $env:PATH.Length) + '%PATH%' }
        $s += "$(ConvertTo-Arg "$k=$v") "
    }
    return $s
}

# Open-Child FILE ARGS DIR ENV CAPTURE: the started process.  With CAPTURE,
# standard output and error are redirected; otherwise the child shares this
# console.  The child inherits this process's environment plus ENV.
function Open-Child([string]$File, [string]$Arguments, [string]$Dir, [hashtable]$Env, [bool]$Capture) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $File
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $Dir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $Capture
    $psi.RedirectStandardError = $Capture
    foreach ($k in $Env.Keys) { $psi.EnvironmentVariables[$k] = [string]$Env[$k] }
    return [System.Diagnostics.Process]::Start($psi)
}

# Invoke-Cmd LINE [DIR] [ENV]: LINE under cmd in DIR, waited for in short
# slices so that an interruption reaches the finally block at once; its exit
# status is the last command's.  A non-zero status stops the run.
function Invoke-Cmd([string]$Line, [string]$Dir = $Root, [hashtable]$Env = @{}) {
    if ($Dry) {
        Write-Line "$(Format-Env $Env)cmd /d /s /c `"$Line`"  [in $Dir]"
        return
    }
    $script:Child = Open-Child $env:ComSpec "/d /s /c `"$Line & call exit %^errorlevel%`"" $Dir $Env $false
    while (-not $Child.WaitForExit(200)) { }
    $rc = $Child.ExitCode
    $script:Child = $null
    if ($rc -ne 0) { Exit-Failure "'$Line' exited $rc" }
}

# Invoke-Capture FILE ARGV [DIR] [-AllowFail]: FILE's standard output; a
# non-zero status stops the run unless -AllowFail, which returns $null.
function Invoke-Capture([string]$File, [string[]]$Argv, [string]$Dir = $Root, [switch]$AllowFail) {
    $p = Open-Child $File (Join-Argument $Argv) $Dir @{} $true
    $err = $p.StandardError.ReadToEndAsync()
    $text = $p.StandardOutput.ReadToEnd()
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) {
        if ($AllowFail) { return $null }
        Exit-Failure "'$File $(Join-Argument $Argv)' exited $($p.ExitCode): $($err.Result.Trim())"
    }
    return $text
}

# Add-Log TEXT: TEXT appended to build.log.
function Add-Log([string]$Text) {
    if (-not $Dry) { [System.IO.File]::AppendAllText($BuildLog, "$Text`n", $Ascii) }
}

# Invoke-Build DIR ENV ARGV: ARGV in DIR, its output appended to build.log.
function Invoke-Build([string]$Dir, [hashtable]$Env, [string[]]$Argv) {
    $cmdline = Join-Argument $Argv -ForCmd
    Add-Log "== $(Format-Env $Env)$cmdline  [in .$($Dir.Substring($Root.Length))]"
    Invoke-Cmd "$cmdline >> $(ConvertTo-Arg $BuildLog -ForCmd) 2>&1" $Dir $Env
}

# ---- the rows file -------------------------------------------------------------------

$Rows = New-Object System.Collections.Generic.List[object]
$Step = "reading $RowsFile"
if (-not $Dry) {
    if (Test-Path -LiteralPath $Out) {
        $entries = @(Get-ChildItem -LiteralPath $Out -Force | Where-Object { $ParamsPath -eq '' -or $_.FullName -ne $ParamsPath })
        if ($entries.Length -gt 0) {
            Write-Err "os_bench.ps1: $Out is not empty; give a new directory"
            exit 2
        }
    }
}

# Main body: every failure, and every interruption, ends in the finally block.
try {
    if (-not $Dry -and -not $Opt.Launch) {
        [void](New-Item -ItemType Directory -Force -Path $Out)
        $FlagOk = $true
    }

    foreach ($line in [System.IO.File]::ReadAllLines($RowsFile)) {
        if ($line -eq '' -or $line.StartsWith('#')) { continue }
        $f = $line.Split("`t")
        if ($f.Length -ne 10) { Exit-Failure "row line has $($f.Length) fields, not 10: $line" }
        for ($k = 0; $k -lt 10; $k++) {
            if ($f[$k] -eq '') { Exit-Failure "empty field in: $line" }
            if ($f[$k] -eq '-') { $f[$k] = '' }
        }
        $r = [pscustomobject]@{
            Group = $f[0]; Row = $f[1]; Os = $f[2]; Kind = $f[3]; Threads = $f[4]
            Dpi = $f[5]; Sink = $f[6]; Inv = $f[7]; Extra = $f[8]; Docs = $f[9]
        }
        if ($r.Os -notin @('both', 'linux', 'windows')) { Exit-Failure "$($r.Row): os must be both, linux or windows" }
        if ($r.Kind -notin @('library', 'deskew', 'process')) { Exit-Failure "$($r.Row): kind must be library, deskew or process" }
        if ($r.Sink -notin @('disk', 'ram', 'none')) { Exit-Failure "$($r.Row): sink must be disk, ram or none" }
        if ($r.Threads -notmatch '^[1-9][0-9]*$' -or $r.Dpi -notmatch '^[1-9][0-9]*$') {
            Exit-Failure "$($r.Row): threads and dpi must be positive integers"
        }
        $Rows.Add($r)
    }

    # Get-RowIndex NAME: the index of row NAME in the rows file.
    function Get-RowIndex([string]$Name) {
        for ($n = 0; $n -lt $Rows.Count; $n++) {
            if ($Rows[$n].Row -eq $Name) { return $n }
        }
        Exit-Failure "no row named $Name in $RowsFile"
    }

    # ---- row selection ------------------------------------------------------------------

    $Step = 'row selection'
    $Selected = New-Object System.Collections.Generic.List[int]
    $OnlyLabel = ''
    if ($Opt.Only -ne '') {
        $n = Get-RowIndex $Opt.Only
        if ($Opt.DefenderExcluded) {
            $excl = @($Rows | Where-Object { $_.Row -eq "$($Opt.Only)-excl" })
            if ($excl.Length -gt 0) { $n = Get-RowIndex "$($Opt.Only)-excl" }
        }
        if ($Rows[$n].Os -eq 'linux') { Exit-Failure "$($Opt.Only) runs only on Linux" }
        if ($Rows[$n].Os -eq 'windows' -and -not $Opt.DefenderExcluded) {
            Exit-Failure "$($Opt.Only) needs -DefenderExcluded"
        }
        $Selected.Add($n)
        $OnlyLabel = $Rows[$n].Row
        if ($Opt.Label -ne '') { $OnlyLabel = $Opt.Label }
    } else {
        $groups = @($Opt.Group)
        if ($groups.Length -eq 0) { $groups = @($Rows | ForEach-Object { $_.Group } | Select-Object -Unique) }
        foreach ($g in $groups) {
            if (-not @($Rows | Where-Object { $_.Group -ceq $g })) { Exit-Failure "no group named $g" }
        }
        for ($n = 0; $n -lt $Rows.Count; $n++) {
            if ($Rows[$n].Os -ne 'both') { continue }
            if ($groups -ccontains $Rows[$n].Group) { $Selected.Add($n) }
        }
    }
    $StartupRow = Get-RowIndex 'S-version'

    # ---- rows that cannot run here --------------------------------------------------------

    function Test-Feature([string]$Name) {
        return ($Opt.GpuFeatures.Split(',') -contains $Name)
    }

    # Get-NvidiaDriver: the driver versions nvidia-smi reports, or nothing.
    function Get-NvidiaDriver {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $smi) { return '' }
        $text = Invoke-Capture $smi.Source @('--query-gpu=driver_version', '--format=csv,noheader') -AllowFail
        if (-not $text) { return '' }
        return ((@($text.Split("`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ }) | Sort-Object -Unique) -join ',')
    }

    # Get-SkipReason N: why row N cannot run on this host, or nothing.
    function Get-SkipReason([int]$N) {
        $extra = " $($Rows[$N].Extra) "
        if ($extra.Contains(' --backend cuda ')) {
            if (-not @($CudaFeatures | Where-Object { Test-Feature $_ })) {
                return "-GpuFeatures $($Opt.GpuFeatures) holds no CUDA feature"
            }
            $major = ((Get-NvidiaDriver) -split '[.,]')[0]
            if ($major -notmatch '^[0-9]+$') { return 'nvidia-smi reports no driver version' }
            if ([int]$major -lt $CudaDriverFloor) {
                return "GPU driver $major is older than $CudaDriverFloor, which CUDA 13 PTX needs"
            }
        }
        if ($extra.Contains(' --backend vulkan ') -and -not (Test-Feature 'vulkan')) {
            return "-GpuFeatures $($Opt.GpuFeatures) lacks vulkan"
        }
        return ''
    }

    # Get-RamRootProblem: why -RamRoot cannot hold the ram rows, or nothing.
    function Get-RamRootProblem {
        $root = $Opt.RamRoot
        if ($root -notmatch '^([A-Za-z]:)\\?$') { return "-RamRoot $root is not a drive root such as R:\" }
        $drive = $Matches[1].ToUpperInvariant()
        $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$drive'"
        if (-not $disk) { return "$drive is not a drive" }
        foreach ($other in @($env:SystemDrive, $Out.Substring(0, 2), $Root.Substring(0, 2))) {
            if ($other.ToUpperInvariant() -eq $drive) { return "$drive also holds the system, DIR or the repository" }
        }
        if ($disk.DriveType -notin @(2, 3, 6)) { return "$drive has drive type $($disk.DriveType), not a local disk" }
        if (-not $disk.Size -or [uint64]$disk.Size -gt [uint64]$RamGibLimit * 1GB) {
            return "$drive holds $($disk.Size) bytes, more than $RamGibLimit GiB"
        }
        $own = @('rr-os-bench', 'System Volume Information', '$RECYCLE.BIN')
        $others = @(Get-ChildItem -LiteralPath "$drive\" -Force | Where-Object { $own -notcontains $_.Name } | ForEach-Object { $_.Name })
        if ($others.Length -gt 0) { return "$drive holds other entries: $(($others | Select-Object -First 5) -join ' ')" }
        $probe = "$drive\.os_bench-probe"
        try {
            [System.IO.File]::WriteAllText($probe, '', $Ascii)
            Remove-Item -LiteralPath $probe -Force
        } catch {
            return "$drive is not writable"
        }
        return ''
    }

    $RunRows = New-Object System.Collections.Generic.List[int]
    $Skips = New-Object System.Collections.Generic.List[string]
    foreach ($n in $Selected) {
        if ($n -eq $StartupRow) { continue }
        $reason = Get-SkipReason $n
        if ($reason -eq '' -and $Rows[$n].Sink -eq 'ram' -and $Opt.RamRoot -eq '') {
            $reason = 'no -RamRoot: the ram rows need a dedicated RAM disk'
        }
        if ($reason -ne '') { $Skips.Add("skipped $($Rows[$n].Row): $reason") } else { $RunRows.Add($n) }
    }

    $RamDir = ''
    if (@($RunRows | Where-Object { $Rows[$_].Sink -eq 'ram' }).Length -gt 0) {
        $RamDir = "$($Opt.RamRoot.Substring(0, 2))\rr-os-bench"
        if (-not $Dry) {
            $Step = 'RAM root check'
            $reason = Get-RamRootProblem
            if ($reason -ne '' -and $Smoke) {
                Write-Err "os_bench.ps1: -Smoke skips the ram rows: $reason"
                $RunRows = [System.Collections.Generic.List[int]]@($RunRows | Where-Object { $Rows[$_].Sink -ne 'ram' })
                $RamDir = ''
            } elseif ($reason -ne '') {
                Exit-Failure $reason 'with the owner''s go-ahead, a RAM disk of at most 4 GiB with nothing else on it'
            }
        }
    }

    $NeedPre = @($RunRows | Where-Object { $Rows[$_].Group -eq 'avx2' }).Length -gt 0
    $NeedGpu = @($RunRows | Where-Object { $Rows[$_].Group -eq 'gpu' }).Length -gt 0

    # ---- checks ---------------------------------------------------------------------------

    $Step = 'fixture check'
    foreach ($n in @(@($StartupRow) + @($RunRows))) {
        foreach ($doc in @($Rows[$n].Docs.Split(' ') | Where-Object { $_ })) {
            $file = $doc -replace ':[0-9]+-[0-9]+$', ''
            if (-not (Test-Path -LiteralPath (Join-Path $Root "tests\fixtures\$file") -PathType Leaf)) {
                Exit-Failure "tests\fixtures\$file is missing (it is gitignored: copy it into this clone)"
            }
        }
    }

    $Step = 'thread check'
    $Cpus = [int](Get-CimInstance Win32_ComputerSystem -Property NumberOfLogicalProcessors).NumberOfLogicalProcessors
    foreach ($n in @(@($StartupRow) + @($RunRows))) {
        $t = [int]$Rows[$n].Threads
        if (($t -eq 1 -and $Cpus -lt 3) -or $t -gt $Cpus) {
            Exit-Failure "$($Rows[$n].Row) needs $t thread(s) on CPUs this host's $Cpus logical CPUs lack"
        }
    }

    if (-not $Dry) {
        $Step = 'build environment check'
        foreach ($v in @('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS')) {
            if ([Environment]::GetEnvironmentVariable($v)) {
                Exit-Failure "$v is set and would replace the repository's -C target-cpu=native" "Remove-Item Env:$v"
            }
        }
    }

    # Get-PowerScheme: the active scheme's GUID and name.
    function Get-PowerScheme {
        $text = Invoke-Capture 'powercfg.exe' @('/getactivescheme') -AllowFail
        if ($text -and $text -match '([0-9a-fA-F-]{36})\s+\((.*)\)') {
            return @{ Guid = $Matches[1].ToLowerInvariant(); Name = $Matches[2].Trim() }
        }
        return @{ Guid = ''; Name = '' }
    }

    if (-not $Smoke -and -not $Dry) {
        $Step = 'machine-state check'
        $scheme = Get-PowerScheme
        if (-not $PowerSchemes.ContainsKey($scheme.Guid) -and $scheme.Name -notin @($PowerSchemes.Values)) {
            Exit-Failure "the active power scheme is '$($scheme.Name)', not High performance or Ultimate Performance" $PowerRemedy
        }
        $p = Open-Child 'git' 'diff --quiet HEAD --' $Root @{} $false
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) {
            Exit-Failure 'the working tree differs from HEAD, whose commit env.tsv records' 'commit or stash the changes'
        }
    }

    # ---- launch ---------------------------------------------------------------------------

    if ($Opt.Launch) {
        $Step = 'launch'
        $forward = New-Object System.Collections.Generic.List[string]
        foreach ($g in $Opt.Group) { $forward.Add('-Group'); $forward.Add($g) }
        foreach ($k in @('Only', 'Label', 'RamRoot')) {
            if ($Opt[$k] -ne '') { $forward.Add("-$k"); $forward.Add($Opt[$k]) }
        }
        $forward.Add('-Reps'); $forward.Add($Opt.Reps)
        $forward.Add('-GpuFeatures'); $forward.Add($Opt.GpuFeatures)
        foreach ($k in @('Smoke', 'DefenderExcluded')) {
            if ($Opt[$k]) { $forward.Add("-$k") }
        }
        $params = Join-Path $Out $ParamsName
        $tr = "powershell -NoProfile -ExecutionPolicy Bypass -File $(ConvertTo-Arg $PSCommandPath) -ParamsFile $(ConvertTo-Arg $params)"
        if ($tr.Length -gt $TaskCommandLimit) {
            Exit-Failure "the task command is $($tr.Length) characters, over schtasks' $TaskCommandLimit" 'a shorter -Out or clone path'
        }
        $user = [string](Get-CimInstance Win32_ComputerSystem -Property UserName).UserName
        if ($user -eq '' -and -not $Dry) {
            Exit-Failure 'no user is logged in at the console; the task runs in that session' 'log in at the console'
        }
        $create = @('/create', '/tn', $TaskName, '/tr', $tr, '/sc', 'once', '/st', '23:59', '/ru', $user, '/it', '/f')
        if ($Opt.DefenderExcluded) { $create += @('/rl', 'highest') }
        $shown = @($create | ForEach-Object { if ($_ -ceq $user) { '<console user>' } else { $_ } })
        if ($Dry) {
            Write-Line "# $params ($($forward.Count) lines):"
            foreach ($a in $forward) { Write-Line "#   $a" }
            Write-Line "# task command: $($tr.Length) of $TaskCommandLimit characters"
            Write-Line "schtasks $(Join-Argument $shown)"
            Write-Line "schtasks /run /tn $TaskName"
            exit 0
        }
        [void](New-Item -ItemType Directory -Force -Path $Out)
        [System.IO.File]::WriteAllText($params, (($forward -join "`n") + "`n"), $Ascii)
        $FlagOk = $true
        Write-Line (Invoke-Capture 'schtasks.exe' $create).Trim()
        Write-Line (Invoke-Capture 'schtasks.exe' @('/run', '/tn', $TaskName)).Trim()
        Write-Line "started task $TaskName; DIR\done.flag or DIR\failed.flag ends it (tests/compare/os_bench_wait.sh)"
        $FlagOk = $false
        $Finished = $true
        exit 0
    }

    # ---- builds -------------------------------------------------------------------------------

    $BuildEnv = @{}
    $nasm = Get-Command nasm -ErrorAction SilentlyContinue
    if (-not $nasm) {
        $nasmDir = Join-Path $env:ProgramFiles 'NASM'
        if (-not (Test-Path -LiteralPath (Join-Path $nasmDir 'nasm.exe'))) {
            $Step = 'build (HEAD)'
            Exit-Failure "nasm is neither on PATH nor in $nasmDir; the AV1 decoder's assembly needs it" 'install NASM'
        }
        $BuildEnv['PATH'] = "$nasmDir;$env:PATH"
    }

    $Step = 'build (HEAD)'
    Invoke-Build $Root $BuildEnv @('cargo', 'build', '--release', '-p', 'rasterrocket-cli')
    Invoke-Build $Root $BuildEnv @('cargo', 'build', '--release', '-p', 'bench', '--bin', 'os_bench')

    # Test-PreCurrent: the pre-AVX2 worktree exists as this run would create
    # it: at HEAD, nothing unstaged, and its index the tree recorded when the
    # revert was applied.  Reusing it keeps the build's fingerprints valid.
    function Test-PreCurrent {
        if (-not (Test-Path -LiteralPath $PreDir) -or -not (Test-Path -LiteralPath $PreStamp)) { return $false }
        $head = Invoke-Capture 'git' @('rev-parse', 'HEAD')
        $preHead = Invoke-Capture 'git' @('-C', $PreDir, 'rev-parse', 'HEAD') -AllowFail
        if (-not $preHead -or $preHead.Trim() -ne $head.Trim()) { return $false }
        if ($null -eq (Invoke-Capture 'git' @('-C', $PreDir, 'diff', '--quiet') -AllowFail)) { return $false }
        $tree = Invoke-Capture 'git' @('-C', $PreDir, 'write-tree') -AllowFail
        return ($tree -and $tree.Trim() -eq ([System.IO.File]::ReadAllText($PreStamp)).Trim())
    }

    if ($NeedPre) {
        $Step = 'build (pre-avx2 worktree)'
        if ($Dry) {
            Write-Line "git worktree add --detach $(ConvertTo-Arg $PreDir) HEAD"
            Write-Line "git -C $(ConvertTo-Arg $PreDir) revert --no-commit $($AvxCommits -join ' ')"
        } elseif (Test-PreCurrent) {
            Add-Log "== reusing $($PreDir.Substring($Root.Length + 1)) (HEAD with $($AvxCommits -join ' ') reverted)"
        } else {
            [void](Invoke-Capture 'git' @('worktree', 'prune'))
            if (Test-Path -LiteralPath $PreStamp) { Remove-Item -LiteralPath $PreStamp -Force }
            if (Test-Path -LiteralPath $PreDir) {
                if ($null -eq (Invoke-Capture 'git' @('worktree', 'remove', '--force', $PreDir) -AllowFail)) {
                    Exit-Failure "$PreDir exists and is not a removable worktree"
                }
            }
            Invoke-Build $Root @{} @('git', 'worktree', 'add', '--detach', $PreDir, 'HEAD')
            try {
                Invoke-Build $Root @{} (@('git', '-C', $PreDir, 'revert', '--no-commit') + $AvxCommits)
            } catch {
                Exit-Failure "reverting $($AvxCommits -join ' ') at HEAD does not apply cleanly; the pre-AVX2 rows cannot run"
            }
            $tree = (Invoke-Capture 'git' @('-C', $PreDir, 'write-tree')).Trim()
            [System.IO.File]::WriteAllText($PreStamp, "$tree`n", $Ascii)
        }
        Invoke-Build $PreDir $BuildEnv @('cargo', 'build', '--release', '-p', 'bench', '--bin', 'os_bench', '--target-dir', $PreTarget)
    }

    if ($NeedGpu) {
        $Step = 'build (gpu features)'
        $gpuEnv = $BuildEnv.Clone()
        $gpuEnv['CUDA_ARCH'] = 'sm_75'
        Invoke-Build $Root $gpuEnv @('cargo', 'build', '--release', '-p', 'rasterrocket-cli',
            '--features', $Opt.GpuFeatures, '--target-dir', $GpuTarget)
    }

    # ---- environment -------------------------------------------------------------------------

    # Get-Probe BLOCK: BLOCK's value as one line, or nothing when it fails.
    function Get-Probe([scriptblock]$Block) {
        try {
            $v = & $Block
            if ($null -eq $v) { return '' }
            return ([string]$v -replace '[\t\r\n]+', ' ').Trim()
        } catch {
            return ''
        }
    }

    if (-not $Dry) {
        $Step = 'environment capture'
        $os = Get-CimInstance Win32_OperatingSystem -Property Caption, BuildNumber, TotalVisibleMemorySize
        $hags = Get-Probe {
            $v = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Name HwSchMode -ErrorAction SilentlyContinue)
            if (-not $v) { 'default' } elseif ($v.HwSchMode -eq 2) { 'on' } elseif ($v.HwSchMode -eq 1) { 'off' } else { "value $($v.HwSchMode)" }
        }
        $envLines = [ordered]@{
            os = 'windows'
            os_build = Get-Probe { $os.Caption }
            kernel = Get-Probe { $os.BuildNumber }
            commit = (Invoke-Capture 'git' @('rev-parse', 'HEAD')).Trim()
            rustc = (Invoke-Capture 'rustc' @('-V')).Trim()
            profile = 'release'
            cpu_model = Get-Probe { (Get-CimInstance Win32_Processor -Property Name | Select-Object -First 1).Name }
            logical_cpus = "$Cpus"
            ram_gib = Get-Probe { ([double]$os.TotalVisibleMemorySize / 1048576).ToString('0.0', $Invariant) }
            power = Get-Probe { (Get-PowerScheme).Name }
            hags = $hags
            defender_rt = Get-Probe { "$((Get-MpComputerStatus).RealTimeProtectionEnabled)".ToLowerInvariant() }
            defender_excl = "$([bool]$Opt.DefenderExcluded)".ToLowerInvariant()
            gpu_driver = Get-Probe { Get-NvidiaDriver }
            cuda = ''
            fs = Get-Probe { (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($Out.Substring(0, 2))'").FileSystem }
            ram_root_fs = ''
            date = [DateTime]::UtcNow.ToString('yyyy-MM-dd', $Invariant)
        }
        if ($NeedGpu -and (Get-SkipReason (Get-RowIndex 'G-cuda-start')) -eq '') {
            $envLines.cuda = Get-Probe {
                $t = Invoke-Capture 'nvcc' @('--version') -AllowFail
                if ($t -match 'release ([0-9.]+)') { $Matches[1] }
            }
        }
        if ($RamDir -ne '') {
            $envLines.ram_root_fs = Get-Probe {
                $d = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($RamDir.Substring(0, 2))'"
                "$($d.FileSystem) $($d.Size)"
            }
        }
        $sb = New-Object System.Text.StringBuilder
        foreach ($k in $envLines.Keys) {
            if ($envLines[$k] -ne '') { [void]$sb.Append("$k`t$($envLines[$k])`n") }
        }
        [System.IO.File]::WriteAllText((Join-Path $Out 'env.tsv'), $sb.ToString(), $Ascii)
        foreach ($name in @('results.tsv', 'pages.tsv', 'harness.log')) {
            [System.IO.File]::WriteAllText((Join-Path $Out $name), '', $Ascii)
        }
        if ($Skips.Count -gt 0) {
            [System.IO.File]::WriteAllText((Join-Path $Out 'skipped.txt'), (($Skips -join "`n") + "`n"), $Ascii)
        }
        if ($RamDir -ne '') {
            if (Test-Path -LiteralPath $RamDir) { Remove-Item -LiteralPath $RamDir -Recurse -Force }
            [void](New-Item -ItemType Directory -Path $RamDir)
            $RamMade = $true
        }
    } else {
        foreach ($s in $Skips) { Write-Line "# $s" }
    }

    # ---- Defender exclusions -------------------------------------------------------------------

    if ($Opt.DefenderExcluded) {
        $Step = 'Defender exclusion'
        if ($Dry) {
            Write-Line "Add-MpPreference -ExclusionPath $(ConvertTo-Arg $Out) -ExclusionProcess rrocket.exe  # those not present; removed at the end"
        } else {
            $pref = Get-MpPreference
            if (@($pref.ExclusionPath) -notcontains $Out) { $ExclPaths = @($Out) }
            if (@($pref.ExclusionProcess) -notcontains 'rrocket.exe') { $ExclProcs = @('rrocket.exe') }
            if ($ExclPaths) { Add-MpPreference -ExclusionPath $ExclPaths }
            if ($ExclProcs) { Add-MpPreference -ExclusionProcess $ExclProcs }
            $pref = Get-MpPreference
            if (@($pref.ExclusionPath) -notcontains $Out -or @($pref.ExclusionProcess) -notcontains 'rrocket.exe') {
                Exit-Failure 'the Defender exclusions are not in effect after adding them' 'run from an elevated session'
            }
        }
    }

    # ---- rows -----------------------------------------------------------------------------------

    $Labels = @{}   # label -> @(row index, variant), for the noise pass.

    # Get-Mask THREADS: the hexadecimal affinity mask of a THREADS-worker row.
    function Get-Mask([int]$Threads) {
        if ($Threads -eq 1) { return '4' }
        return ([uint64][Math]::Pow(2, $Threads) - 1).ToString('X')
    }

    # Invoke-Harness DEST THREADS BENCH ARGV: one pinned harness call, its
    # standard output appended to DEST and its standard error to harness.log.
    function Invoke-Harness([string]$Dest, [int]$Threads, [string]$Bench, [string[]]$Argv) {
        $line = "start `"`" /b /wait /affinity $(Get-Mask $Threads) $(Join-Argument (@($Bench) + $Argv) -ForCmd)" +
            " >> $(ConvertTo-Arg $Dest -ForCmd) 2>> $(ConvertTo-Arg $HarnessLog -ForCmd)"
        try {
            Invoke-Cmd $line $Root @{ RAYON_NUM_THREADS = "$Threads" }
        } catch {
            if ($_.Exception.Data['os_bench']) {
                $tail = @(Get-Content -LiteralPath $HarnessLog -Tail 5 -ErrorAction SilentlyContinue)
                Exit-Failure "harness exited non-zero; harness.log ends:`n$($tail -join "`n")"
            }
            throw
        }
    }

    # Invoke-Row N DEST REPS VARIANT [DOC]: row N on build VARIANT (head|pre),
    # its samples appended to DEST; with DOC, only the document of that name.
    function Invoke-Row([int]$N, [string]$Dest, [int]$RowReps, [string]$Variant, [string]$OnlyDoc = '') {
        $r = $Rows[$N]
        $label = $r.Row
        if ($OnlyLabel -ne '' -and $N -eq $Selected[0] -and $N -ne $StartupRow) { $label = $OnlyLabel }
        $bench = $HeadBench
        $rr = $HeadRr
        if ($Variant -eq 'pre') { $label += '-pre'; $bench = $PreBench }
        if ($r.Group -eq 'gpu') { $rr = $GpuRr }
        $script:Labels[$label] = @($N, $Variant)
        $script:Step = "row $label"
        $docs = New-Object System.Collections.Generic.List[string]
        foreach ($doc in @($r.Docs.Split(' ') | Where-Object { $_ })) {
            if ($OnlyDoc -ne '' -and ($doc -replace ':[0-9]+-[0-9]+$', '') -ne $OnlyDoc) { continue }
            $docs.Add("tests\fixtures\$doc")
            if ($Smoke) { break }
        }
        if ($r.Docs -ne '' -and $docs.Count -eq 0) { Exit-Failure "$label has no document $OnlyDoc" }
        $extra = @($r.Extra.Split(' ') | Where-Object { $_ })
        $a = New-Object System.Collections.Generic.List[string]
        if ($r.Kind -eq 'process') {
            $a.AddRange([string[]]@('process', '--group', $r.Group, '--row', $label, '--rrocket', $rr,
                    '--dpi', $r.Dpi, '--threads', $r.Threads, '--reps', "$RowReps"))
            if ($r.Inv -ne '') { $a.AddRange([string[]]@('--invocations', $r.Inv)) }
            switch ($r.Sink) {
                'disk' { $a.AddRange([string[]]@('--disk', (Join-Path $Scratch $label), '--expect-pages')) }
                'ram' { $a.AddRange([string[]]@('--ram', $RamDir, '--spill', (Join-Path $Scratch "spill-$label"))) }
                'none' { $a.Add('--none') }
            }
            $a.AddRange([string[]]@($docs))
            if ($extra.Length -gt 0) { $a.Add('--'); $a.AddRange([string[]]$extra) }
        } else {
            $a.AddRange([string[]]@('library', '--group', $r.Group, '--row', $label, '--dpi', $r.Dpi,
                    '--threads', $r.Threads, '--reps', "$RowReps"))
            if ($r.Kind -eq 'deskew') { $a.Add('--deskew') }
            foreach ($w in $extra) {
                if ($w -eq '--page-times') {
                    # Re-runs keep the first run's per-page times.
                    if ($Dest -eq (Join-Path $Out 'results.tsv')) { $a.AddRange([string[]]@('--page-times', (Join-Path $Out 'pages.tsv'))) }
                } else {
                    $a.Add($w)
                }
            }
            $a.AddRange([string[]]@($docs))
        }
        Invoke-Harness $Dest ([int]$r.Threads) $bench $a.ToArray()
        if (-not $Dry -and (Test-Path -LiteralPath $Scratch)) { Remove-Item -LiteralPath $Scratch -Recurse -Force }
    }

    $Results = Join-Path $Out 'results.tsv'
    Invoke-Row $StartupRow $Results $Reps 'head'
    foreach ($n in $RunRows) {
        Invoke-Row $n $Results $Reps 'head'
        if ($Rows[$n].Group -eq 'avx2') { Invoke-Row $n $Results $Reps 'pre' }
    }

    $Step = 'noise pass'
    $Rerun = Join-Path $Out 'rerun.tsv'
    if ($Dry) {
        Write-Line "$(ConvertTo-Arg $HeadBench) noisy $(ConvertTo-Arg $Out)  # each ROW<TAB>DOC listed: that row and document again, --reps 5, >> $(ConvertTo-Arg $Rerun)"
    } else {
        $noisy = Invoke-Capture $HeadBench @('noisy', $Out)
        foreach ($l in @($noisy.Split("`n") | ForEach-Object { $_.TrimEnd("`r") } | Where-Object { $_ })) {
            $label, $doc = $l.Split("`t", 2)
            if (-not $Labels.ContainsKey($label)) { Exit-Failure "os_bench noisy lists $label, which this run did not run" }
            Invoke-Row $Labels[$label][0] $Rerun 5 $Labels[$label][1] $doc
        }
    }

    $Step = 'finish'
    $Finished = $true
} catch {
    $e = $_.Exception
    if ($e.Data['os_bench']) {
        $msg = $e.Message
    } else {
        $msg = "${Step}: line $($_.InvocationInfo.ScriptLineNumber): $($e.Message)"
    }
    if ($Step -like 'build*' -and (Test-Path -LiteralPath $BuildLog)) {
        Get-Content -LiteralPath $BuildLog -Tail 20 | ForEach-Object { Write-Err $_ }
    }
    $msg = Hide-Path $msg
    Write-Err "os_bench.ps1: $msg"
    Write-Flag $msg
    exit 1
} finally {
    # The flag comes first, and each cleanup step runs alone: a failing step
    # neither skips the others nor lets done.flag claim a clean run.
    if ($FlagOk -and -not $Finished) {
        Write-Flag (Hide-Path "${Step}: interrupted")
    }
    if ($Child -and -not $Child.HasExited) {
        Invoke-Cleanup 'stopping the child process tree' {
            [void](Invoke-Capture 'taskkill.exe' @('/pid', "$($Child.Id)", '/t', '/f') -AllowFail)
        }
    }
    if ($RamMade) {
        Invoke-Cleanup "removing $RamDir" { Remove-Item -LiteralPath $RamDir -Recurse -Force }
    }
    if (-not $Dry -and (Test-Path -LiteralPath $Scratch)) {
        Invoke-Cleanup 'removing DIR\scratch' { Remove-Item -LiteralPath $Scratch -Recurse -Force }
    }
    if ($ExclPaths) {
        Invoke-Cleanup 'removing the Defender path exclusion' { Remove-MpPreference -ExclusionPath $ExclPaths }
    }
    if ($ExclProcs) {
        Invoke-Cleanup 'removing the Defender process exclusion' { Remove-MpPreference -ExclusionProcess $ExclProcs }
    }
    if ($Finished -and $FlagOk -and -not $CleanupFailed) {
        [System.IO.File]::WriteAllText((Join-Path $Out 'done.flag'), '', $Ascii)
    }
}
if ($CleanupFailed) { exit 1 }
