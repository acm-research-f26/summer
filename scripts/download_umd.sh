#!/usr/bin/env bash
# Fetch the UMD (Uterine Myoma MRI Dataset) archive from figshare into data/.
#
#   doi:10.6084/m9.figshare.23541312  (v3, CC BY 4.0)  UMD.zip, 4.76 GB
#
# Resumable: re-running after an interrupted download continues where it left
# off. The md5 is checked before extraction, so a truncated file is never
# unzipped. Extracts to data/UMD/, which is what config/default.yaml's
# data.root ("data/UMD/UMD") expects.
#
# Env:
#   KEEP_ZIP=0   delete UMD.zip after a verified extraction (frees 4.8 GB)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
ZIP="$DATA/UMD.zip"
DEST="$DATA/UMD"
URL="https://ndownloader.figshare.com/files/44111183"
MD5="f52985246c363a7fb9c8f6ae33dcde84"
KEEP_ZIP="${KEEP_ZIP:-1}"

mkdir -p "$DATA"

if [ -f "$ZIP" ]; then
  echo "resuming existing $ZIP ($(du -h "$ZIP" | cut -f1) so far)"
fi
echo "downloading UMD.zip (4.76 GB) from figshare ..."
curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$ZIP" "$URL"

echo "verifying md5 ..."
GOT="$(md5 -q "$ZIP" 2>/dev/null || md5sum "$ZIP" | cut -d' ' -f1)"
if [ "$GOT" != "$MD5" ]; then
  echo "md5 MISMATCH: expected $MD5, got $GOT" >&2
  echo "the archive is corrupt or incomplete; delete $ZIP and re-run" >&2
  exit 1
fi
echo "md5 OK ($GOT)"

echo "extracting to $DEST ..."
mkdir -p "$DEST"
unzip -q -o "$ZIP" -x '__MACOSX/*' -d "$DEST"

if [ "$KEEP_ZIP" = "0" ]; then
  rm -f "$ZIP"
  echo "removed $ZIP"
fi

# count under data.root, not $DEST: the archive nests a second UMD/ level, and
# a plain find over $DEST also picks up __MACOSX resource-fork stubs.
echo "done. patients: $(ls -d "$DEST"/UMD/UMD_* | wc -l | tr -d ' ')" \
     "masks: $(find "$DEST/UMD" -name '*_se[gq].nii.gz' -not -name '._*' | wc -l | tr -d ' ')"
