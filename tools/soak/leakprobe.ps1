param([string]$Mode = "none", [int]$Minutes = 20)
Set-Location (Join-Path $PSScriptRoot "..\..")
$dir = "target\leakprobe-$Mode"
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
$p = Start-Process -FilePath "target\release\examples\leakprobe.exe" -ArgumentList @("$Minutes", $Mode, $dir) -PassThru -NoNewWindow -RedirectStandardOutput "target\leakprobe-$Mode.log"
$null = $p.Handle
$samples = @()
while (-not $p.HasExited) {
    Start-Sleep -Seconds 2
    $p.Refresh()
    if (-not $p.HasExited) { $samples += [math]::Round($p.PrivateMemorySize64 / 1MB, 1) }
}
$n = $samples.Count
"mode=$Mode samples=$n first=$($samples[[math]::Min(2, $n-1)]) MB mid=$($samples[[int]($n/2)]) MB last=$($samples[$n-1]) MB"
Get-Content "target\leakprobe-$Mode.log" -Tail 1
Remove-Item -Recurse -Force $dir
