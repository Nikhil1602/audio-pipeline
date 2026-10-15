# Soak runner: records for $Seconds with processing, keeps an inaudible noise playing so the
# loopback stream carries real audio, and logs the recorder's resources every minute.
param([int]$Seconds = 14400, [string]$Out = "target\soak1")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$exe = Join-Path $PWD "target\release\sdctl.exe"
$csv = "$Out.resources.csv"
"utc,elapsed_s,working_set_mb,private_mb,cpu_s,handles,threads,player_restarts" | Out-File -Encoding utf8 $csv

$rec = Start-Process -FilePath $exe -PassThru -NoNewWindow `
    -ArgumentList @("record", "--seconds", "$Seconds", "--process", "--agc", "--restart-every", "1800", "--out", $Out) `
    -RedirectStandardOutput "$Out.log" -RedirectStandardError "$Out.err.log"
$null = $rec.Handle  # needed for ExitCode with -NoNewWindow
$t0 = Get-Date
$player = $null
$restarts = 0
$next = $t0
while (-not $rec.HasExited) {
    if ($null -eq $player -or $player.HasExited) {
        $player = Start-Process -FilePath $exe -PassThru -NoNewWindow `
            -ArgumentList @("play", "target\soak_noise.wav") `
            -RedirectStandardOutput "$Out.player.log" -RedirectStandardError "$Out.player.err.log"
        $restarts++
    }
    if ((Get-Date) -ge $next) {
        $rec.Refresh()
        $el = [int]((Get-Date) - $t0).TotalSeconds
        "{0},{1},{2:F1},{3:F1},{4:F1},{5},{6},{7}" -f (Get-Date).ToUniversalTime().ToString("s"), $el,
            ($rec.WorkingSet64 / 1MB), ($rec.PrivateMemorySize64 / 1MB), $rec.TotalProcessorTime.TotalSeconds,
            $rec.HandleCount, $rec.Threads.Count, $restarts | Out-File -Append -Encoding utf8 $csv
        $next = $next.AddSeconds(60)
    }
    Start-Sleep -Milliseconds 500
}
if ($null -ne $player -and -not $player.HasExited) { Stop-Process -Id $player.Id -Force }
"recorder exit code: $($rec.ExitCode)" | Out-File -Append -Encoding utf8 "$Out.log"
