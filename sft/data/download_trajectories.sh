#!/usr/bin/env bash
set -euo pipefail

# Output directory for extracted trajectories
OUTDIR="."
ZIPFILE="$OUTDIR/trajectories.zip"

# Release asset URL
ZIP_URL="https://github.com/eggrollius/UniAgent-AFM/releases/download/trajectories/20250522_sweagent_claude-4-sonnet-20250514.zip"

echo "Downloading SWE-bench trajectories..."

# Download the zip file
curl -L "$ZIP_URL" -o "$ZIPFILE"

# Extract contents into OUTDIR
unzip -q -o "$ZIPFILE" -d "$OUTDIR"

# Remove the zip file (optional — comment out if you want to keep it)
rm "$ZIPFILE"

echo "Done. Files are available in: $OUTDIR/"
