#!/bin/bash
set -euo pipefail

app_path="dist/EPUB Audiobook Maker.app"

if [[ ! -d "$app_path" ]]; then
  echo "PyInstaller app bundle not found: $app_path" >&2
  exit 1
fi

mkdir -p artifacts
hdiutil create \
  -volname "EPUB Audiobook Maker" \
  -srcfolder "$app_path" \
  -ov \
  -format UDZO \
  "artifacts/EPUB-Audiobook-Maker-macos-${MAC_ARCH}.dmg"
