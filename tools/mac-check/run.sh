#!/bin/bash
# Downloads the SonicDiar Mac check kit and starts it. Paste this into Terminal:
#
#   curl -fsSL https://github.com/Nikhil1602/sonicdiar/releases/download/mac-check/run.sh | bash
#
# The kit goes to ~/sonicdiar-check (about 15 MB, plus models it downloads when asked).
set -eu
url="https://github.com/Nikhil1602/sonicdiar/releases/download/mac-check/sonicdiar-mac-check.zip"
dir="$HOME/sonicdiar-check"
echo "Downloading the SonicDiar Mac check ..."
mkdir -p "$dir"
curl -fsSL "$url" -o "$dir/kit.zip"
cd "$dir"
unzip -qo kit.zip
rm -f kit.zip
exec bash ./check.sh
