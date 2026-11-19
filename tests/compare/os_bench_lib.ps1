# Helpers of tests/compare/os_bench.ps1, which dot-sources this file: output
# and redaction, the failure flag, child processes and builds, machine-state
# probes, the RAM-root check, Defender exclusions and the scheduled-task
# launch.  The functions read the driver's script variables ($Root, $Out,
# $Dry, $Opt, $Step and the state the driver documents next to them).

# ---- output and redaction -------------------------------------------------------

# Initialize-Redaction: the console session's DOMAIN\user and its two parts,
# read once so that Hide-Path can remove them from every message.
function Initialize-Redaction {
    $script:ConsoleUser = ''
    $script:Redactions = @()
    try {
        $script:ConsoleUser = [string](Get-CimInstance Win32_ComputerSystem -Property UserName).UserName
    } catch {
        $script:ConsoleUser = ''
    }
    if ($ConsoleUser -ne '') {
        $parts = @($ConsoleUser.Split('\') | Where-Object { $_.Length -ge 2 })
        $script:Redactions = @(@($ConsoleUser) + $parts)
    }
}

# Hide-Path TEXT: TEXT with the repository path removed, every
# X:\Users\NAME directory shown as ~, and the console user and machine name
# shown as <user>.
function Hide-Path([string]$Text) {
    $t = [regex]::Replace($Text, [regex]::Escape("$Root\"), '', 'IgnoreCase')
    $t = [regex]::Replace($t, '[A-Za-z]:[\\/]Users[\\/][^\\/\s"'']+', '~', 'IgnoreCase')
    foreach ($r in $Redactions) {
        $t = [regex]::Replace($t, "(?<![A-Za-z0-9_])$([regex]::Escape($r))(?![A-Za-z0-9_])", '<user>', 'IgnoreCase')
    }
    return $t
}

function Write-Err([string]$Text) {
    [Console]::Error.WriteLine((Hide-Path $Text))
}

function Write-Line([string]$Text) {
    [Console]::Out.WriteLine((Hide-Path $Text))
}

function Show-Usage([int]$Code) {
    $print = $false
    foreach ($line in [System.IO.File]::ReadAllLines($DriverPath)) {
        if ($line -like '# Usage:*') { $print = $true }
        if ($line -like '# Rows whose*') { break }
        if ($print) { [Console]::Error.WriteLine(($line -replace '^# ?', '')) }
    }
    exit $Code
}

# Exit-Usage TEXT: a usage error, written to failed.flag when DIR is already
# the run's own (a scheduled-task run), then exit 2.
function Exit-Usage([string]$Text) {
    Write-Err "os_bench.ps1: $Text"
    Write-Flag "argument checks: $Text"
    if ($TaskMode) { Unregister-BenchTask }
    Show-Usage 2
}

# ---- arguments ---------------------------------------------------------------------

# Read-Option ARGV: the options as a hashtable.  Named parameters are parsed
# here rather than by the binder, which refuses a repeated -Group.
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

# ---- failure reporting ----------------------------------------------------------

# Write-Flag MESSAGE: failed.flag holding MESSAGE (redacted), once DIR is the
# run's own and unless one exists: the first report is kept.
function Write-Flag([string]$Message) {
    if (-not $FlagOk) { return }
    $flag = Join-Path $Out 'failed.flag'
    if (-not [System.IO.File]::Exists($flag)) {
        [System.IO.File]::WriteAllText($flag, "$(Hide-Path $Message)`n", $Ascii)
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
        $msg = "cleanup: ${What}: $($_.Exception.Message)"
        Write-Err "os_bench.ps1: $msg"
        Write-Flag $msg
    }
}

# ---- small helpers -------------------------------------------------------------------

# Get-Drive PATH: the upper-case drive (X:) of an absolute PATH.
function Get-Drive([string]$Path) {
    return $Path.Substring(0, 2).ToUpperInvariant()
}

# Get-DocFile DOC: the file name of a rows-file document, its range dropped.
function Get-DocFile([string]$Doc) {
    return ($Doc -replace ':[0-9]+-[0-9]+$', '')
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
# slices so that an interruption reaches the finally block at once.  The
# appended `call exit %^errorlevel%` makes cmd's status the last command's:
# `start /b /wait` alone returns 0 whatever its program returned.  A non-zero
# status stops the run.
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

# Test-Command FILE ARGV [DIR]: whether FILE exits 0.
function Test-Command([string]$File, [string[]]$Argv, [string]$Dir = $Root) {
    return ($null -ne (Invoke-Capture $File $Argv $Dir -AllowFail))
}

# Add-Log FILE TEXT: TEXT appended as one line to FILE.
function Add-Log([string]$File, [string]$Text) {
    if (-not $Dry) { [System.IO.File]::AppendAllText($File, "$Text`n", $Ascii) }
}

# Invoke-Build DIR ENV ARGV: ARGV in DIR, its output appended to build.log.
function Invoke-Build([string]$Dir, [hashtable]$Env, [string[]]$Argv) {
    $cmdline = Join-Argument $Argv -ForCmd
    Add-Log $BuildLog "== $(Format-Env $Env)$cmdline  [in .$($Dir.Substring($Root.Length))]"
    Invoke-Cmd "$cmdline >> $(ConvertTo-Arg $BuildLog -ForCmd) 2>&1" $Dir $Env
}

# Get-Mask THREADS: the hexadecimal affinity mask of a THREADS-worker row.
function Get-Mask([int]$Threads) {
    if ($Threads -eq 1) { return '4' }
    return ([uint64][Math]::Pow(2, $Threads) - 1).ToString('X')
}

# Invoke-Harness LABEL DEST THREADS BENCH ARGV: one pinned harness call, its
# standard output appended to DEST and its standard error to harness.log
# after a `== row LABEL` marker.  On failure, the lines after the last marker
# go into the message.
function Invoke-Harness([string]$Label, [string]$Dest, [int]$Threads, [string]$Bench, [string[]]$Argv) {
    $line = "start `"`" /b /wait /affinity $(Get-Mask $Threads) $(Join-Argument (@($Bench) + $Argv) -ForCmd)" +
        " >> $(ConvertTo-Arg $Dest -ForCmd) 2>> $(ConvertTo-Arg $HarnessLog -ForCmd)"
    Add-Log $HarnessLog "== row $Label"
    try {
        Invoke-Cmd $line $Root @{ RAYON_NUM_THREADS = "$Threads" }
    } catch {
        if (-not $_.Exception.Data['os_bench']) { throw }
        $log = @([System.IO.File]::ReadAllLines($HarnessLog))
        $from = 0
        for ($i = 0; $i -lt $log.Length; $i++) {
            if ($log[$i].StartsWith('== row ')) { $from = $i + 1 }
        }
        $tail = @('(nothing)')
        if ($from -lt $log.Length) {
            $tail = @($log[$from..($log.Length - 1)] | Where-Object { $_ -ne '' } | Select-Object -Last 10)
        }
        Exit-Failure "harness exited non-zero; its standard error:`n$($tail -join "`n")"
    }
}

# ---- probes ------------------------------------------------------------------------

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

# Get-NvidiaDriver: the driver versions nvidia-smi reports, or nothing.
function Get-NvidiaDriver {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return '' }
    $text = Invoke-Capture $smi.Source @('--query-gpu=driver_version', '--format=csv,noheader') -AllowFail
    if (-not $text) { return '' }
    return ((@($text.Split("`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ }) | Sort-Object -Unique) -join ',')
}

# Get-PowerScheme: the active scheme's GUID and name.
function Get-PowerScheme {
    $text = Invoke-Capture 'powercfg.exe' @('/getactivescheme') -AllowFail
    if ($text -and $text -match '([0-9a-fA-F-]{36})\s+\((.*)\)') {
        return @{ Guid = $Matches[1].ToLowerInvariant(); Name = $Matches[2].Trim() }
    }
    return @{ Guid = ''; Name = '' }
}

# Get-Priority: this process's priority class, which the harness inherits.
function Get-Priority {
    return [string][System.Diagnostics.Process]::GetCurrentProcess().PriorityClass
}

# Get-RamRootProblem: why -RamRoot cannot hold the ram rows, or nothing.  The
# volume must identify itself as a RAM disk: drive type 6, or a fixed disk
# labelled $RamLabel.  It must also be neither the system drive nor the drive
# of DIR or the repository, at most $RamGibLimit GiB, empty but for this
# driver's directory and the file system's own entries, and writable.
function Get-RamRootProblem {
    $root = $Opt.RamRoot
    if ($root -notmatch '^[A-Za-z]:\\?$') { return "-RamRoot $root is not a drive root such as R:\" }
    $drive = Get-Drive $root
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$drive'"
    if (-not $disk) { return "$drive is not a drive" }
    foreach ($other in @($env:SystemDrive, $Out, $Root)) {
        if ((Get-Drive $other) -eq $drive) { return "$drive also holds the system, DIR or the repository" }
    }
    $isRam = $disk.DriveType -eq 6 -or ($disk.DriveType -eq 3 -and [string]$disk.VolumeName -eq $RamLabel)
    if (-not $isRam) {
        return "$drive (drive type $($disk.DriveType), label '$($disk.VolumeName)') is not a RAM disk: drive type 6, or a fixed disk labelled $RamLabel"
    }
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

# ---- Defender exclusions --------------------------------------------------------------

# Add-DefenderExclusion: the exclusions of DIR and rrocket.exe that are not
# present already; each is recorded for removal only once its add returned.
function Add-DefenderExclusion {
    if ($Dry) {
        Write-Line "Add-MpPreference -ExclusionPath $(ConvertTo-Arg $Out) -ExclusionProcess rrocket.exe  # those not present; removed at the end"
        return
    }
    $pref = Get-MpPreference
    if (@($pref.ExclusionPath) -notcontains $Out) {
        Add-MpPreference -ExclusionPath $Out
        $script:ExclPaths = @($Out)
    }
    if (@($pref.ExclusionProcess) -notcontains 'rrocket.exe') {
        Add-MpPreference -ExclusionProcess 'rrocket.exe'
        $script:ExclProcs = @('rrocket.exe')
    }
    $pref = Get-MpPreference
    if (@($pref.ExclusionPath) -notcontains $Out -or @($pref.ExclusionProcess) -notcontains 'rrocket.exe') {
        Exit-Failure 'the Defender exclusions are not in effect after adding them' 'run from an elevated session'
    }
}

# ---- scheduled-task launch ---------------------------------------------------------------

# Unregister-BenchTask: the scheduled task removed, when it exists.
function Unregister-BenchTask {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
}

# Invoke-Launch: write the options to DIR\launch-params.txt and start this
# driver on them as the scheduled task, in the console user's session, at
# normal priority (the scheduler's default, 7, is below normal) and with the
# highest run level (so that the run can unregister its own task, and
# -DefenderExcluded can add exclusions); then exit 0.
function Invoke-Launch {
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
    $argument = "-NoProfile -ExecutionPolicy Bypass -File $(ConvertTo-Arg $DriverPath) -ParamsFile $(ConvertTo-Arg $params)"
    if ($ConsoleUser -eq '' -and -not $Dry) {
        Exit-Failure 'no user is logged in at the console; the task runs in that session' 'log in at the console'
    }
    if ($Dry) {
        Write-Line "# $params ($($forward.Count) lines):"
        foreach ($a in $forward) { Write-Line "#   $a" }
        Write-Line "Unregister-ScheduledTask -TaskName $TaskName  # when it exists"
        Write-Line "Register-ScheduledTask -TaskName $TaskName -Action (powershell.exe $argument) -Principal (<user>, Interactive, Highest) -Settings (Priority 4, MultipleInstances IgnoreNew, no time limit)"
        Write-Line "Start-ScheduledTask -TaskName $TaskName"
        exit 0
    }
    Unregister-BenchTask
    [void](New-Item -ItemType Directory -Force -Path $Out)
    [System.IO.File]::WriteAllText($params, (($forward -join "`n") + "`n"), $Ascii)
    $script:FlagOk = $true
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument
    $principal = New-ScheduledTaskPrincipal -UserId $ConsoleUser -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -Priority 4 -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    [void](Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings)
    Start-ScheduledTask -TaskName $TaskName
    Write-Line "started task $TaskName; DIR\done.flag or DIR\failed.flag ends it (tests/compare/os_bench_wait.sh)"
    $script:FlagOk = $false
    $script:Finished = $true
    exit 0
}
