SonicDiar: Mac check
=========================

Thank you for helping! This tests whether the SonicDiar records a Mac's microphone
and system audio correctly. It takes about 10 minutes.

You need: a Mac (Apple silicon or Intel) on macOS 14.2 or newer, speakers or headphones,
and an internet connection. (On macOS 12 or 13 the microphone part works, but not system
audio.) AirPods or wired headphones are useful for the
optional device-change step.

1. Open Terminal (Spotlight: press Cmd+Space, type Terminal, press Enter).
2. Paste this line and press Enter:

   curl -fsSL https://github.com/Nikhil1602/sonicdiar/releases/download/mac-check/run.sh | bash

3. Follow the instructions on screen. macOS will ask to allow Terminal to record system
   audio and to use the microphone: please click Allow for both.
4. At the end, send back the zip file it puts on your Desktop
   (sonicdiar-check-<date>.zip).

Privacy: the results contain device names, test numbers and logs. Your recordings are only
included if you say yes at the end. If you try the transcription step, the text it heard is
in the results. Nothing is sent anywhere by the script; you send the zip yourself.

Clean up afterwards (optional): delete the folder "sonicdiar-check" in your home
folder and the results folder on the Desktop. In System Settings > Privacy & Security you
can turn off Terminal's microphone and system audio access again.
