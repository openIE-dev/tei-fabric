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
  dst="$boards_dir/$id.glb"; raw="$tmp/$id-raw.glb"

  # Convert STEP → raw glTF. Local OpenCascade (cadquery) is the robust path
  # with no gateway cap (Zoo's sync conversion 504s on heavy board assemblies
  # — confirmed; it's synchronous-only with no pollable async for these).
  # Coarse tessellation keeps the mesh web-sized.
  if ! python3 "$here/step2gltf.py" "$step" "$raw" >/dev/null 2>&1 || [ ! -s "$raw" ]; then
    echo "  ✗ $id: STEP→glTF failed (need: pip install cadquery)"; continue
  fi

  # Web-optimize: weld + simplify + dedup + prune, NO compression (so no
  # runtime Draco/meshopt decoder is needed). Turns 20–70 MB raw meshes into
  # ~1–3 MB web glb. Falls back to the raw mesh if gltf-transform is absent.
  if npx --yes @gltf-transform/cli@4.4.0 optimize "$raw" "$dst" --compress false --simplify-error 0.001 >/dev/null 2>&1 && [ -s "$dst" ]; then
    echo "  ✓ $id.glb ($(wc -c < "$dst") bytes, optimized)"
  else
    cp "$raw" "$dst"; echo "  ✓ $id.glb ($(wc -c < "$dst") bytes, raw — install gltf-transform to shrink)"
  fi
done
echo "done. BOARD view loads /boards/<id>.glb automatically when present."
