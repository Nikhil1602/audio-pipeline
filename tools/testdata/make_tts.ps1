# Generates local speech test material with the Windows built-in voices (not committed:
# bench/data/ is git-ignored). Output: 16-bit mono WAV at 48 kHz.
#   far_en.wav  - "Microsoft David Desktop" reading text A (the remote talker)
#   near_en.wav - "Microsoft Zira Desktop" reading text B (the local talker)
param([string]$OutDir = "bench\data")

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$farText = @"
Good morning everyone, thanks for joining the weekly planning call. Let us start with the
release schedule. The first build is ready for testing, and the team has fixed most of the
issues reported last week. We still need to confirm the deployment window with operations.
After that, we will review the budget for the next quarter, and finally we will go through
the open questions from the customer meeting on Tuesday. If anyone has an urgent item,
please raise it now so we can make time for it at the end of the call.
"@

$nearText = @"
Sorry to interrupt, I have a quick question about the testing plan. Are we running the
performance tests on the new hardware, or on the old servers? I also wanted to mention that
the documentation is almost finished, and I can share a draft with everyone by Thursday
afternoon. Please send me your comments before Friday.
"@

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($item in @(@("Microsoft David Desktop", $farText, "far_en.wav"), @("Microsoft Zira Desktop", $nearText, "near_en.wav"))) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($item[0])
    $path = Join-Path (Resolve-Path $OutDir) $item[2]
    $s.SetOutputToWaveFile($path, $fmt)
    $s.Speak($item[1])
    $s.Dispose()
    Write-Output ("wrote {0}" -f $path)
}
