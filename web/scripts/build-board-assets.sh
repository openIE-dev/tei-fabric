#!/usr/bin/env bash
# build-board-assets.sh — produce Studio's BOARD 3D models from vendor CAD.
#
# Accuracy path: each board's `cad_step` in public/boards/manifest.json is
# the VENDOR's own published STEP file. We convert it to web glTF (.glb)
# with the authenticated `zoo` CLI (Zoo/KittyCAD's CAD engine — the same
# one behind the openie-cad MCP). No renders, no guesses: the 3D is the
# manufacturer's CAD.
#
# One-time setup:   zoo auth login
# Run:              web/scripts/build-board-assets.sh   (from anywhere)
#
# Output: web/public/boards/<id>.glb  (served at /boards/<id>.glb)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
boards_dir="$here/../public/boards"
manifest="$boards_dir/manifest.json"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

command -v zoo >/dev/null || { echo "need the 'zoo' CLI (brew install zoo)"; exit 1; }
command -v jq  >/dev/null || { echo "need 'jq'"; exit 1; }
zoo auth status >/dev/null 2>&1 || { echo "run 'zoo auth login' first"; exit 1; }

ids=$(jq -r '.boards | to_entries[] | select(.value.cad_step != null) | .key' "$manifest")
[ -z "$ids" ] && { echo "no boards with a cad_step URL yet — add verified vendor STEP URLs to manifest.json"; exit 0; }

for id in $ids; do
  step_url=$(jq -r ".boards[\"$id\"].cad_step" "$manifest")
  member=$(jq -r ".boards[\"$id\"].cad_step_member // empty" "$manifest")
  echo "→ $id: $step_url"
  src="$tmp/$id.src"
  curl -fsSL "$step_url" -o "$src"
  step="$tmp/$id.step"
  if file "$src" | grep -qi zip; then
    [ -n "$member" ] || { echo "  zip needs cad_step_member in manifest"; continue; }
    unzip -o "$src" "$member" -d "$tmp" >/dev/null
    mv "$tmp/$member" "$step"
  else
    mv "$src" "$step"
  fi
  dst="$boards_dir/$id.glb"

  # Path 1 (fast): Zoo cloud conversion. Works instantly for light files and
  # auto-async for very large ones, but its synchronous endpoint is Cloudflare
  # gateway-capped (~100s) — mid-size-heavy board assemblies (e.g. the full
  # Pico STEP) 504 here. So we try it, then fall back to local conversion.
  out="$tmp/$id-out"; mkdir -p "$out"
  if zoo file convert --output-format=gltf "$step" "$out/" >/dev/null 2>&1 \
     && glb=$(find "$out" -maxdepth 1 \( -name '*.glb' -o -name '*.gltf' \) | head -1) \
     && [ -n "$glb" ]; then
    cp "$glb" "$dst"
    echo "  ✓ (zoo) public/boards/$id.glb ($(wc -c < "$dst") bytes)"
    continue
  fi

  # Path 2 (robust, no gateway): local OpenCascade conversion via cadquery.
  echo "  zoo path unavailable for $id (heavy STEP / gateway timeout) — converting locally…"
  if python3 "$here/step2gltf.py" "$step" "$dst" >/dev/null 2>&1 && [ -s "$dst" ]; then
    echo "  ✓ (local) public/boards/$id.glb ($(wc -c < "$dst") bytes)"
  else
    echo "  ✗ $id: both paths failed (for local, run: pip install cadquery)"
  fi
done
echo "done. BOARD view loads /boards/<id>.glb automatically when present."
