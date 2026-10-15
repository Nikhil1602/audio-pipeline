param([int]$Minutes = 30, [string]$Arena = "1", [string]$Pattern = "1", [string]$Opt = "1", [string]$Pre = "0")
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:SD_ORT_DYLIB = Join-Path $PWD "models\onnxruntime.dll"
$p = Start-Process -FilePath "target\release\examples\vadmem.exe" -ArgumentList @("$Minutes", $Arena, $Pattern, $Opt, $Pre) -PassThru -NoNewWindow -RedirectStandardOutput "target\vadmem.log" -RedirectStandardError "target\vadmem.err.log"
$null = $p.Handle
$samples = @()
while (-not $p.HasExited) {
    Start-Sleep -Milliseconds 500
    $p.Refresh()
    if (-not $p.HasExited) { $samples += [math]::Round($p.PrivateMemorySize64 / 1MB, 1) }
}
$n = $samples.Count
"arena=$Arena pattern=$Pattern opt=$Opt prealloc=$Pre minutes=$Minutes samples=$n early=$($samples[[math]::Min(3, $n-1)]) MB mid=$($samples[[int]($n/2)]) MB last=$($samples[$n-1]) MB exit=$($p.ExitCode)"
Get-Content target\vadmem.err.log -Tail 2
